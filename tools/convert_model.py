"""Developer-only export of the bundled model; see MODEL.md and asset licenses.

Architecture based on BasicSR SRVGGNetCompact, Apache-2.0.
Pretrained weights: Real-ESRGAN v0.2.5.0, BSD-3-Clause.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class SRVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList([nn.Conv2d(3,64,3,1,1), nn.PReLU(64)])
        for _ in range(32):
            self.body.extend([nn.Conv2d(64,64,3,1,1), nn.PReLU(64)])
        self.body.append(nn.Conv2d(64,48,3,1,1))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return F.pixel_shuffle(out,4) + F.interpolate(x,scale_factor=4,mode='nearest')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    expected = '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != expected:
        raise ValueError('Weights do not match the official general-x4v3 release.')
    network = SRVGG().eval()
    network.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True)['params'])
    torch.manual_seed(42)
    sample = torch.rand(1,3,32,40)
    torch.onnx.export(network, sample, str(args.output), input_names=['image'], output_names=['enhanced'],
        dynamic_axes={'image':{2:'height',3:'width'}, 'enhanced':{2:'height4',3:'width4'}},
        opset_version=17, dynamo=False)
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    out = session.run(None, {'image': sample.numpy()})[0]
    with torch.no_grad():
        error = float(np.max(np.abs(out - network(sample).numpy())))
    assert error < 1e-5, f'Conversion mismatch: {error}'
    print('Verified maximum error:', error)
    print('SHA-256:', hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
