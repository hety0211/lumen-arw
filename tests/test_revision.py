import copy
import threading
from pathlib import Path
import cv2
import numpy as np
import pytest
from lumen import model, engine, curves, white_balance, retouch


def test_curve_all_brightness_values_and_shape_preservation():
    points = [[i / 255, (i / 255)**.7] for i in range(256)]
    r = model.recipe()
    r['curves']['RGB'] = points
    assert len(model.validate(r)['curves']['RGB']) == 256
    x = np.linspace(0, 1, 4097)
    y = curves.evaluate(points, x)
    assert np.all(np.diff(y) >= 0) and y[0] == 0 and y[-1] == 1
    np.testing.assert_allclose(curves.evaluate(points, np.arange(256)/255), np.asarray(points)[:,1], atol=1e-12)
    rgb = np.repeat(np.linspace(0,1,256,dtype=np.float32)[None,:,None],3,axis=2)
    np.testing.assert_allclose(engine.apply_curves(rgb, r['curves'], 'smooth')[0,:,0], np.asarray(points)[:,1], atol=.0002)


def test_old_recipe_keeps_linear_curve_appearance():
    old = model.recipe()
    old['version'] = 2
    old.pop('curve_mode')
    old['curves']['R'] = [[0,0],[.3,.55],[1,1]]
    new = model.validate(old)
    assert new['curve_mode'] == 'linear'
    rgb = np.random.default_rng(4).random((30,40,3), dtype=np.float32)
    np.testing.assert_array_equal(engine.apply_curves(rgb, old['curves']), engine.apply_curves(rgb,new['curves'],new['curve_mode']))


def test_camera_kelvin_metadata_and_honest_auto_fallback():
    exact = white_balance.from_metadata({'ColorTemperature': 5200}, 4800)
    assert exact == dict(camera_kelvin=5200.,kelvin=5200.,estimated=False)
    auto = white_balance.from_metadata({'ColorTemperature': 0}, 4540)
    assert auto['estimated'] and auto['camera_kelvin'] == 4540
    assert white_balance.from_metadata({})['kelvin'] is None
    np.testing.assert_array_equal(white_balance.gains(exact), np.ones(3))
    exact['kelvin'] = 7200
    gains = white_balance.gains(exact)
    assert gains[0] > 1 and gains[2] < 1


def test_default_camera_white_balance_is_pixel_exact():
    r = model.recipe()
    src = np.random.default_rng(11).random((40,60,3),dtype=np.float32)
    expected = engine.process(src,r)
    r['white_balance'] = white_balance.from_metadata({'ColorTemperature': 5500})
    np.testing.assert_array_equal(engine.process(src,r),expected)
    r['white_balance']['kelvin'] = 7800
    assert not np.allclose(engine.process(src,r),expected)


def operation(kind, points=None, **kwargs):
    return dict(kind=kind, points=points or [[.5,.5]], radius=.06, enabled=True,
                feather=30,opacity=100,**kwargs)


def test_healing_removes_dust_preserves_precision_and_outside():
    src = np.full((120,160,3),[.312345,.456789,.567891],np.float32)
    damaged = src.copy()
    damaged[57:63,77:83] = .01
    healed = retouch.apply(damaged,[operation('heal')])
    np.testing.assert_allclose(healed[59:61,79:81],src[59:61,79:81],atol=1e-4)
    np.testing.assert_array_equal(healed[:45],damaged[:45])
    np.testing.assert_array_equal(damaged[57:63,77:83],np.full((6,6,3),.01,np.float32))
    assert healed.dtype == np.float32


def test_clone_alignment_opacity_boundary_and_disabled():
    yy,xx = np.mgrid[0:101,0:101]
    src = np.stack([xx/100,yy/100,np.full_like(xx,.4,dtype=float)],axis=2).astype(np.float32)
    op = operation('clone',offset=[-.25,-.25])
    out = retouch.apply(src,[op])
    np.testing.assert_allclose(out[50,50],src[25,25],atol=1e-6)
    op['opacity'] = 50
    np.testing.assert_allclose(retouch.apply(src,[op])[50,50],(src[25,25]+src[50,50])/2,atol=1e-6)
    op['offset'] = [-1.,-1.]
    np.testing.assert_array_equal(retouch.apply(src,[op]),src)
    op['enabled'] = False
    assert retouch.apply(src,[op]) is src


def test_retouch_project_snapshot_preset_and_full_size_coordinates(tmp_path):
    r = model.recipe()
    r['retouch'] = [operation('clone',offset=[-.2,0]),operation('heal',[[.7,.7]])]
    r['white_balance'] = white_balance.from_metadata({'ColorTemperature': 4300})
    r['white_balance']['kelvin'] = 5700
    output = tmp_path/'修复.lumen'
    model.save_project(output,tmp_path/'source.ARW',r,[dict(name='修复后',edits=r)])
    _, loaded,snapshots = model.load_project(output,True)
    assert loaded == r == snapshots[0]['edits']
    look = model.builtin_presets()[0]['look']
    result = model.apply_look(r,look)
    assert result['retouch'] == r['retouch'] and result['white_balance'] == r['white_balance']
    for shape in [(101,201),(201,401)]:
        src = np.tile(np.linspace(0,1,shape[1],dtype=np.float32)[None,:,None],(shape[0],1,3))
        out = retouch.apply(src,[r['retouch'][0]])
        np.testing.assert_allclose(out[shape[0]//2,shape[1]//2],.3,atol=.001)


@pytest.mark.parametrize('bad', [float('nan'),0,99999])
def test_invalid_kelvin_rejected(bad):
    r = model.recipe()
    r['white_balance'].update(camera_kelvin=5500,kelvin=bad)
    with pytest.raises(ValueError):model.validate(r)


def test_real_bundled_neural_model_changes_pixels_and_honors_cancel():
    src = np.random.default_rng(9).random((24,31,3),dtype=np.float32)
    progress=[]
    out,name = engine.super_resolve(src,2,':builtin:',False,lambda a,b: progress.append((a,b)))
    assert out.shape == (48,62,3) and 'Real-ESRGAN' in name and 'CPUExecutionProvider' in name
    assert progress[-1] == (1,1) and np.isfinite(out).all()
    classical,_ = engine.super_resolve(src,2)
    assert np.mean(np.abs(out-classical)) > .005
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):engine.super_resolve(src,2,':builtin:',False,cancel=event)


def test_real_neural_tiled_output_has_no_join_seam():
    import onnxruntime as ort
    src=np.random.default_rng(3).random((24,205,3),dtype=np.float32)
    out,_=engine.super_resolve(src,4,':builtin:',False)
    path=Path(engine.__file__).resolve().parents[1]/'assets/models/realesr-general-x4v3.onnx'
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    direct=np.clip(session.run(None,{'image':src.transpose(2,0,1)[None].copy()})[0][0].transpose(1,2,0),0,1)
    np.testing.assert_allclose(out,direct,atol=2e-5)
