"""Convert hash-verified official NAFNet / Real-ESRGAN weights. Build-only torch."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
import onnxruntime as ort
from restoration_arch import NAFNet,RRDBNet,DRUNet

SPECS={
    'denoise':('NAFNet-SIDD-width32.pth','89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4','nafnet-sidd.onnx'),
    'super':('RealESRGAN_x4plus.pth','4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1','realesrgan-x4plus.onnx'),
    'drunet':('drunet_color.pth','479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4','drunet-color.onnx')}

def convert(kind,weights_dir,destination):
    filename,digest,outname=SPECS[kind];path=weights_dir/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Unexpected weight checksum'
    torch.set_num_threads(8)
    net=NAFNet(width=32,middle_blk_num=12,enc_blk_nums=[2,2,4,8],dec_blk_nums=[2,2,2,2]) if kind=='denoise' else RRDBNet(3,3,scale=4,num_feat=64,num_block=23,num_grow_ch=32)
    if kind=='drunet':net=DRUNet()
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint.get('params_ema',checkpoint.get('params',checkpoint)),strict=True);net.eval()
    example=torch.rand(1,3,32,48)
    inputs=(example,torch.full((1,1,1,1),15/255)) if kind=='drunet' else example
    output=destination/outname;destination.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(net,inputs,str(output),opset_version=17,dynamo=False,input_names=['image','sigma'] if kind=='drunet' else ['image'],output_names=['output'],dynamic_axes={'image':{2:'height',3:'width'},'output':{2:'output_height',3:'output_width'}})
    options=ort.SessionOptions();options.intra_op_num_threads=8
    session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider'])
    errors=[]
    for h,w in [(32,48),(35,43)]:
        sample=torch.from_numpy(np.random.default_rng(19).random((1,3,h,w),dtype=np.float32))
        with torch.no_grad():expected=(net(sample,inputs[1]) if kind=='drunet' else net(sample)).numpy()
        feed={'image':sample.numpy()}
        if kind=='drunet':feed['sigma']=inputs[1].numpy()
        actual=session.run(None,feed)[0]
        error=float(np.max(np.abs(actual-expected)));assert error<1e-4,error;errors.append(error)
    urls={
        'super':'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'drunet':'https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth',
        'denoise':'https://drive.usercontent.google.com/download?id=1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ&export=download&confirm=t'}
    return dict(file=outname,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,conversion_error=max(errors),parameters=sum(p.numel() for p in net.parameters()),weights=filename,weights_sha256=digest,source=urls[kind],license='BSD-3-Clause / Apache-2.0' if kind=='super' else 'MIT')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('weights',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    records=[]
    for kind in SPECS:
        record=convert(kind,args.weights,args.output);records.append(record);print(json.dumps(record),flush=True)
    (args.output/'v12-models.json').write_text(json.dumps(records,indent=2),encoding='utf8')
