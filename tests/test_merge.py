import threading
from pathlib import Path
import numpy as np
import cv2
import pytest
from lumen import merge,model
from lumen.merge_dialog import reference_index,reserve_result,run_records


def texture(h=240,w=360):
    rng=np.random.default_rng(120)
    image=cv2.GaussianBlur(rng.random((h,w,3),dtype=np.float32),(0,0),.65)
    for _ in range(65):
        x,y=int(rng.integers(0,w)),int(rng.integers(0,h))
        cv2.circle(image,(x,y),int(rng.integers(3,16)),tuple(map(float,rng.random(3))),-1)
    return image


def panorama_views(h=300,w=420):
    world=texture(800,2000);focal=w*.92
    yy,xx=np.mgrid[:h,:w];rays=np.stack([(xx-w/2)/focal,(yy-h/2)/focal,np.ones_like(xx)],axis=2)
    views=[]
    for degrees in (-25,0,25):
        angle=np.deg2rad(degrees);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        direction=rays@rotation.T
        mx=(np.arctan2(direction[...,0],direction[...,2])+np.pi)/(2*np.pi)*world.shape[1]
        my=(np.arctan2(direction[...,1],np.sqrt(direction[...,0]**2+direction[...,2]**2))+np.pi/2)/np.pi*world.shape[0]
        views.append(cv2.remap(world,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR))
    return views


def test_reference_is_largest_not_filename_and_collision_safe(tmp_path):
    assert reference_index([dict(width=500,height=400),dict(width=800,height=600),dict(width=600,height=400)])==1
    assert reference_index([dict(width=10,height=10)]*3)==1
    assert reference_index([dict(width=10,height=10)]*3,0)==0
    old=tmp_path/'原片-HDR堆栈.dng';old.write_bytes(b'keep')
    result=reserve_result(tmp_path,'原片','hdr')
    assert result.name=='原片-HDR堆栈-2.dng' and old.read_bytes()==b'keep'


def test_stack_alignment_recovers_translation_and_common_crop():
    image=texture();matrix=np.float32([[1,0,9],[0,1,-7]])
    moved=cv2.warpAffine(image,matrix,(360,240),borderMode=cv2.BORDER_REFLECT)
    aligned,valid=merge.aligned_stack([image,moved],0)
    roi=merge.inner_crop(valid);a,b,c,d=roi
    assert np.mean(np.abs(aligned[0][b:d,a:c]-aligned[1][b:d,a:c]))<.012
    assert 300<c-a<=351 and 200<d-b<=234


def test_mertens_is_used_and_deghost_levels_are_ordered(monkeypatch):
    image=texture();images=[np.clip(image*f,0,1) for f in (.5,1,1.8)]
    calls=[];original=cv2.createMergeMertens
    def create(*args):calls.append(args);return original(*args)
    monkeypatch.setattr(cv2,'createMergeMertens',create)
    output,info=merge.merge_images(images,'hdr',1,align=False)
    assert calls==[(1.,1.,1.)] and info['algorithm']=='Mertens exposure fusion'
    assert output.shape==image.shape and np.isfinite(output).all()
    assert np.std(output)>.07 and not np.allclose(output,image)
    moving=[image.copy(),image.copy(),image.copy()];moving[0][80:140,110:180]=.84
    masks=[merge.ghost_mask(moving,1,level) for level in ('low','medium','high')]
    assert 0<masks[0].sum()<=masks[1].sum()<=masks[2].sum()
    cleaned=merge.fuse_hdr(moving,1,'high')
    assert np.abs(cleaned[90:130,125:165]-image[90:130,125:165]).mean()<.02


def test_focus_stack_recovers_sharp_regions():
    clear=texture();blur=cv2.GaussianBlur(clear,(0,0),4)
    a=blur.copy();a[:,:180]=clear[:,:180]
    b=blur.copy();b[:,180:]=clear[:,180:]
    fused,_=merge.merge_images([a,b],'focus',align=False)
    assert np.mean((fused-clear)**2)<min(np.mean((a-clear)**2),np.mean((b-clear)**2))*.12


def test_hdr_motion_replacement_has_no_reference_brightness_patch():
    from lumen.engine import to_linear,to_srgb
    base=texture(300,420)*.5+.25
    static=[np.clip(to_srgb(to_linear(base)*ev),0,1) for ev in (.3,1,3)]
    moving=[a.copy() for a in static];moving[0][80:160,120:210]=(.08,.4,.18)
    expected=merge.fuse_hdr(static,1,'high');actual=merge.fuse_hdr(moving,1,'high')
    assert np.mean(np.abs(actual[95:145,135:195]-expected[95:145,135:195]))<.04


def test_real_spherical_panorama_projection_crop_and_cap():
    views=panorama_views();output,info=merge.merge_images(views,'panorama',1)
    assert output.shape[1]>views[0].shape[1]*1.4 and output.shape[0]>200
    assert info['pixel_limit']==200_000_000 and np.isfinite(output).all()
    # The overlap should reconstruct the middle camera's content, rather than
    # simply concatenating independent photographs.
    assert output.std()>.07 and output.min()>.001
    repeated,_=merge.merge_images(views,'panorama',1)
    assert repeated.shape==output.shape and np.max(np.abs(repeated-output))<.001
    k=np.array([[10000.,0,10000],[0,10000,7500],[0,0,1.]])
    _,bounds,scale=merge.bounded_rois([(15000,20000)]*2,[k,k],[np.eye(3),np.eye(3)],20000)
    assert bounds[2]*bounds[3]<=200_000_000 and scale<20000


def test_cancel_and_unmatched_photos_fail_cleanly():
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):merge.merge_images([texture(),texture()],'hdr',cancel=cancel)
    with pytest.raises(ValueError,match='尺寸'):merge.merge_images([texture(),texture(220,300)],'hdr',align=False)
    with pytest.raises(ValueError,match='全景匹配'):merge.merge_images([np.zeros((160,240,3),np.float32)]*2,'panorama')


def test_complete_file_pipeline_keeps_metadata_and_largest_reference(tmp_path):
    from PIL import Image
    base=texture();records=[]
    for name,shape in [('999',(270,180)),('001',(360,240))]:
        p=tmp_path/(name+'.png');Image.fromarray(np.uint8(cv2.resize(base,shape)*255)).save(p)
        records.append(dict(path=str(p),edits=model.recipe(),initialized=False))
    result=run_records(records,'focus',preview=False)
    assert Path(result['source']).stem=='001'
    assert result['image'].shape[1]>300
