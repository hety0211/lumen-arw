import copy
import threading
import numpy as np
import pytest
from lumen import model,engine,watermark,photo_metadata,denoise,selection
from lumen.ai_dialog import reserve_copy


@pytest.mark.parametrize('suffix',['.CRW','.CR2','.CR3','.NEF','.NRW','.RAF','.RW2','.RAW','.ARW'])
def test_requested_raw_formats(suffix):
    assert suffix.lower() in engine.RAW_EXTENSIONS
    assert '*'+suffix.lower() in engine.PHOTO_FILTER


def test_watermark_reads_capture_metadata_and_unknowns():
    p=photo_metadata.from_tags(dict(Make='Canon',Model='EOS R',LensModel='SIGMA 35mm F1.4',
                                   DateTimeOriginal='2025:03:02 14:05:06',FNumber=2.8,ExposureTime=.002,ISO=800))
    assert p['date']=='2025-03-02 14:05:06' and p['shutter']=='1/500 s'
    assert p['aperture']=='f/2.8' and p['iso']=='ISO 800' and p['lens_brand']=='SIGMA'
    assert all('None' not in line for line in watermark.lines(watermark.defaults(),{}))
    assert '拍摄时间未记录' in watermark.lines(watermark.defaults(),{})


@pytest.mark.parametrize('side',['top','bottom','left','right'])
@pytest.mark.parametrize('preset',list(watermark.PRESETS))
def test_borders_preserve_original_float_pixels(side,preset):
    rgb=np.random.default_rng(4).random((180,260,3),dtype=np.float32)
    settings=dict(watermark.defaults(),enabled=True,preset=preset,sides=[side])
    output=watermark.apply(rgb,settings,dict(body='Canon EOS R',lens='RF 24-105mm F4 L IS USM',date='2025-01-01 10:00:00'))
    pad=round(180*.14);y=pad if side=='top' else 0;x=pad if side=='left' else 0
    assert (output.shape[1],output.shape[0])==watermark.dimensions(rgb.shape,settings)
    np.testing.assert_array_equal(output[y:y+180,x:x+260],rgb)
    assert output.dtype==np.float32 and np.isfinite(output).all()


def test_watermark_project_roundtrip_and_old_migration(tmp_path):
    r=model.recipe();r['watermark'].update(enabled=True,sides=['left','bottom'],preset='travel')
    model.save_project(tmp_path/'photo.lumen',tmp_path/'source.RAF',r)
    assert model.load_project(tmp_path/'photo.lumen')[1]['watermark']==r['watermark']
    for version in range(1,5):
        old=copy.deepcopy(r);old['version']=version;old.pop('watermark')
        assert model.validate(old)['watermark']==watermark.defaults()
    assert model.apply_look(r,model.extract_look(model.recipe()))['watermark']==r['watermark']
    for bad in [dict(r['watermark'],size=float('nan')),dict(r['watermark'],sides=['unknown']),dict(r['watermark'],camera_logo='broken')]:
        with pytest.raises((ValueError,TypeError)):watermark.validate(bad)


def test_custom_transparent_logo_survives_recipe_and_renders(tmp_path):
    import base64,io
    from PIL import Image
    emblem=Image.new('RGBA',(40,40),(250,20,10,0))
    emblem.paste((250,20,10,255),(10,10,30,30))
    buffer=io.BytesIO();emblem.save(buffer,format='PNG')
    r=model.recipe();r['watermark'].update(enabled=True,fields=[],size=25.,
        camera_logo=base64.b64encode(buffer.getvalue()).decode('ascii'))
    model.save_project(tmp_path/'logo.lumen',tmp_path/'original.CR3',r)
    settings=model.load_project(tmp_path/'logo.lumen')[1]['watermark']
    rgb=np.full((240,320,3),.37,np.float32)
    result=watermark.apply(rgb,settings,{})
    np.testing.assert_array_equal(result[:240],rgb)
    border=result[240:]
    assert np.any((border[:,:,0]>.9)&(border[:,:,1]<.1))
    np.testing.assert_allclose(border[-1,-1],np.array(watermark.PRESETS['gallery'][1])/255,atol=1e-7)


def test_metadata_survives_generated_linear_dng(tmp_path):
    path=tmp_path/'photo-去杂色.dng'
    photo=photo_metadata.from_tags(dict(Make='Nikon',Model='Z 6',LensModel='NIKKOR Z 50mm f/1.8 S',ISO=640))
    rgb=np.full((64,90,3),.4,np.float32)
    engine.export_image(path,rgb,photo=photo,provenance=dict(operation='denoise'))
    decoded,info=engine.load_image(path,None)
    assert info['photo']==photo and not info['raw']
    np.testing.assert_allclose(engine.to_srgb(decoded),rgb,atol=.0001)


def test_copy_collision_never_overwrites_original(tmp_path):
    first=reserve_copy(tmp_path,'photo','super');first.write_bytes(b'preserve')
    second=reserve_copy(tmp_path,'photo','super')
    assert first.name=='photo-增强.dng' and second.name=='photo-增强-2.dng'
    assert first.read_bytes()==b'preserve'
    assert reserve_copy(tmp_path,'photo','denoise').name=='photo-去杂色.dng'


def test_real_ffdnet_reduces_noise_and_has_no_tile_seams():
    y,x=np.mgrid[0:301,0:307]
    clean=np.stack([.2+.5*x/306,.2+.5*y/300,np.full_like(x,.4,dtype=float)],axis=2).astype(np.float32)
    noisy=np.clip(clean+np.random.default_rng(33).normal(0,25/255,clean.shape).astype(np.float32),0,1)
    ticks=[];output,backend=denoise.process(noisy,100/3,False,lambda n,t:ticks.append((n,t)))
    assert 'CPU' in backend and ticks[-1][0]==ticks[-1][1]
    assert np.mean((output-clean)**2)<np.mean((noisy-clean)**2)*.2
    whole=np.pad(noisy,((0,1),(0,1),(0,0)),mode='edge')
    sigma=denoise.noise_level(noisy)*(100/3)/35
    assert abs(denoise.noise_level(noisy)-25/255)<.012
    result=denoise.session(False).run(None,dict(image=whole.transpose(2,0,1)[None].copy(),sigma=np.full((1,1,1,1),sigma,np.float32)))[0][0].transpose(1,2,0)
    np.testing.assert_allclose(output,np.clip(result[:301,:307],0,1),atol=2e-5)
    # A preview must use the full photo's noise estimate, with matching results
    # away from the patch boundary (the neural receptive field).
    patch=noisy[32:224,32:224]
    preview,_=denoise.process(patch,100/3,False,noise_reference=denoise.noise_level(noisy))
    np.testing.assert_allclose(preview[32:-32,32:-32],output[64:192,64:192],atol=2e-5)
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):denoise.process(noisy,cancel=cancel)
    cancel.clear()
    with pytest.raises(InterruptedError):denoise.process(noisy,progress=lambda n,t:cancel.set(),cancel=cancel)


def test_subject_is_background_inverse_and_depth_is_finite():
    rgb=np.random.default_rng(2).random((100,140,3),dtype=np.float32)
    subject,_=selection.automatic(rgb,'subject');background,_=selection.automatic(rgb,'background')
    np.testing.assert_allclose(subject+background,1,atol=1e-6)
    near,provider=selection.automatic(rgb,'foreground')
    assert near.shape==rgb.shape[:2] and np.isfinite(near).all() and 'CPU' in provider
    for kind in ('subject','foreground'):
        r=model.recipe();mask=model.new_mask(kind,1);mask['raster']=selection.encode(near);r['masks']=[mask]
        assert model.validate(r)['masks'][0]['kind']==kind
