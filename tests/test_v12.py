import copy,gc,threading
from pathlib import Path
import numpy as np
import pytest
from lumen import model,engine,exposure_curve,performance,large_image,restoration,watermark


def test_thread_ceiling_and_cpu_provider_configuration():
    assert performance.MAX_THREADS==32 and 1<=performance.THREADS<=32
    opts=performance.session_options()
    assert opts.intra_op_num_threads==performance.THREADS and opts.inter_op_num_threads==1


@pytest.mark.parametrize('key',('exposure','contrast',*exposure_curve.KEYS))
def test_exposure_plot_matches_actual_tonal_pixels(key):
    a=model.adjustments();a[key]=.8 if key=='exposure' else 35
    axis=np.linspace(0,1,1025,dtype=np.float32)
    source=np.repeat(engine.to_linear(axis)[None,:,None],3,axis=2)
    actual=engine.Backend('cpu').tonal(source,a)[0,:,0]
    np.testing.assert_allclose(exposure_curve.evaluate(a,axis),actual,atol=3e-7)


@pytest.mark.parametrize('x',(.02,.08,.2,.32,.5,.65,.8,.92,1.))
def test_drag_any_tone_solves_linked_sliders_without_extra_curve(x):
    a=model.adjustments();target=x-.03 if x>.85 else x+.025
    updated=exposure_curve.drag(a,x,target)
    assert all(-100<=updated[k]<=100 for k in exposure_curve.KEYS)
    assert abs(float(exposure_curve.evaluate(updated,x))-target)<.001
    assert updated['exposure']==0 and updated['contrast']==0
    assert any(updated[k]!=0 for k in exposure_curve.KEYS)


def test_shift_drag_is_global_exposure_and_extremes_are_bounded():
    a=model.adjustments();b=exposure_curve.drag(a,.5,.65,True)
    assert b['exposure']>0 and all(b[k]==0 for k in exposure_curve.KEYS)
    assert abs(float(exposure_curve.evaluate(b,.5))-.65)<.002
    for x,y in ((0,1),(1,0),(.1,-5),(.5,10)):
        b=exposure_curve.drag(a,x,y)
        assert np.isfinite(list(b.values())).all()
        assert all(-100<=b[k]<=100 for k in exposure_curve.KEYS)


def test_limit_is_inclusive_and_applies_to_borders_and_ai(monkeypatch):
    assert engine.MAX_EXPORT_PIXELS==400_000_000
    large_image.validate_size((20000,20000,3))
    with pytest.raises(ValueError,match='4 亿'):large_image.validate_size((20001,20000,3))
    class Shape:shape=(10001,10000,3)
    with pytest.raises(ValueError,match='4 亿'):engine.super_resolve(Shape(),2,':quality:')
    monkeypatch.setattr(large_image,'MAX_PIXELS',400)
    with pytest.raises(ValueError):watermark.apply(np.zeros((20,20,3),np.float32),dict(watermark.defaults(),enabled=True),{})


def test_disk_mapping_lifetime_and_streamed_pipeline_match(tmp_path,monkeypatch):
    monkeypatch.setattr(large_image,'MAP_BYTES',1024)
    source=large_image.allocate((137,93,3));path=Path(source.filename)
    source[:]=np.random.default_rng(10).random(source.shape,dtype=np.float32)*.7
    edits=model.recipe();edits['adjustments'].update(exposure=.3,shadows=15,highlights=-20)
    edits['hsl'][1]=[4,-3,5];edits['curves']['R']=[[0,0],[.5,.54],[1,1]]
    edits['crop']=[.1,.12,.9,.93];edits['rotation']=1
    expected=engine.process(np.asarray(source),edits,_stream=False)
    output=engine.process(source,edits)
    np.testing.assert_allclose(output,expected,atol=2e-6)
    assert path.exists()
    view=source[:3];del source;gc.collect();assert path.exists()
    del view;gc.collect();assert not path.exists()
    target=tmp_path/'large.dng';engine.export_image(target,output)
    restored,info=engine.load_image(target,None)
    np.testing.assert_allclose(engine.to_srgb(restored),output,atol=.00012)
    preview,metadata=engine.load_image(target,40)
    assert max(preview.shape[:2])==40 and metadata['width']==output.shape[1]


def test_feathered_tiling_odd_dimensions_is_exact_for_identity(monkeypatch):
    class Session:
        def run(self,_,inputs):return [inputs['image']]
        def get_providers(self):return ['CPUExecutionProvider']
    monkeypatch.setattr(restoration,'session',lambda *a:Session())
    rgb=np.random.default_rng(14).random((333,517,3),dtype=np.float32)
    out,_=restoration.tiled(rgb,'denoise',cuda=False)
    np.testing.assert_allclose(out,rgb,atol=2e-7)
    region,_=restoration.tiled(rgb,'denoise',cuda=False,region=(181,173,407,311))
    np.testing.assert_allclose(region,out[173:311,181:407],atol=2e-7)
    cancel=threading.Event()
    with pytest.raises(InterruptedError):restoration.tiled(rgb,'denoise',progress=lambda n,t:cancel.set(),cancel=cancel)


def test_actual_new_models_and_noise_quality():
    y,x=np.mgrid[:81,:95]
    clean=np.stack([.2+.4*x/95,.3+.2*y/81,np.full_like(x,.4,dtype=float)],axis=2).astype(np.float32)
    noise=np.random.default_rng(45).normal(0,1,clean.shape).astype(np.float32)*np.sqrt(.001*clean+.0002)
    noisy=np.clip(clean+noise,0,1)
    output,name=restoration.denoise(noisy,100,False)
    assert 'NAFNet' in name and np.mean((output-clean)**2)<np.mean(noise**2)*.5
    dru,name=restoration.drunet_denoise(noisy,35,False)
    assert 'DRUNet' in name and np.mean((dru-clean)**2)<np.mean(noise**2)*.5
    preview,_=restoration.drunet_denoise(noisy,35,False,region=(17,19,64,60))
    np.testing.assert_allclose(preview,dru[19:60,17:64],atol=2e-6)
    result,name=restoration.super_resolution(clean[:24,:31],2,False)
    assert result.shape==(48,62,3) and np.isfinite(result).all() and 'RRDB' in name
    preview,_=restoration.super_resolution(clean[:24,:31],2,False,region=(7,6,23,18))
    np.testing.assert_allclose(preview,result[12:36,14:46],atol=2e-6)
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):restoration.denoise(noisy,cancel=cancel)
