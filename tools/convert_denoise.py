"""Convert MIT-licensed KAIR FFDNet color weights to dynamic ONNX (even input sizes)."""
from pathlib import Path
import argparse
import torch
import numpy as np
import onnxruntime as ort


class FFDNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Conv2d(13, 96, 3, padding=1), torch.nn.ReLU()]
        for _ in range(10):
            layers += [torch.nn.Conv2d(96, 96, 3, padding=1), torch.nn.ReLU()]
        layers += [torch.nn.Conv2d(96, 12, 3, padding=1)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, image, sigma):
        x = torch.nn.functional.pixel_unshuffle(image, 2)
        noise = torch.ones_like(x[:, :1]) * sigma
        return torch.nn.functional.pixel_shuffle(self.model(torch.cat([x, noise], 1)), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    model = FFDNet().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    torch.manual_seed(11)
    x, sigma = torch.rand(1, 3, 64, 80), torch.full((1, 1, 1, 1), 25/255)
    torch.onnx.export(model, (x, sigma), str(args.output), opset_version=17, dynamo=False,
                      input_names=['image', 'sigma'], output_names=['clean'],
                      dynamic_axes={'image': {2: 'height', 3: 'width'}, 'clean': {2: 'height', 3: 'width'}})
    sess = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    result = sess.run(None, {'image': x.numpy(), 'sigma': sigma.numpy()})[0]
    with torch.no_grad(): expected = model(x, sigma).numpy()
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-4)
    print('Maximum conversion error:', np.abs(result-expected).max())


if __name__ == '__main__': main()
