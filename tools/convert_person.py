"""Export the official TorchVision person class to the app's offline ONNX format."""
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
)


class Person(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
        ).eval()

    def forward(self, image):
        return self.model(image)['out'].softmax(1)[:, 15:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    model = Person().eval()
    sample = torch.rand(1, 3, 384, 384)
    torch.onnx.export(model, sample, str(args.output), input_names=['image'],
                      output_names=['person'], opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'image': sample.numpy()})[0]
    with torch.no_grad():
        expected = model(sample).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    print(f'Saved {args.output}; maximum absolute error: {np.max(np.abs(actual-expected)):.8g}')


if __name__ == '__main__':
    main()
