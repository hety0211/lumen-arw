# 模型来源与转换

## 1.3.0 GPU 逐像素图

1.3.0 不新增神经网络模型，九个 ONNX 模型及其校验值保持不变。`lumen/gpu_graphs.py` 在运行时生成三张不含权重的逐像素计算图（`tonal`、`color`、`fused`），用于 DirectML 显影；`lumen/onnx_graph.py` 直接写出 ONNX protobuf，因此运行和构建都不需要 `onnx` 包。图采用 opset 13、通道在后（1 × H × W × 3）的 float32 输入。

- `tonal`：白平衡增益、曝光、阴影／高光／黑色／白色与对比度，公式与 NumPy 版一致。
- `color`：饱和度／自然饱和度、八色 HSL（1° 查找表线性插值，节点均在整数角度上，因此与 `numpy.interp` 相同）、RGB 与单通道曲线（4097 点查找表，与平滑曲线的采样轴相同；折线模式重采样误差约 2×10⁻⁵）、黑白与三段色彩分级。
- `fused`：`tonal` 后接 `color`，未启用空间细节工具时一次完成。

CPU 路径仍是参考实现；`tests/test_v13_pipeline.py` 以 ONNX Runtime CPU 执行这些图并与 NumPy 结果比较。`python tools/build_tonal_dml.py 文件夹` 可导出三张图供 Netron 等工具查看。

## 1.2.1 照片合成

三种合成均使用已随软件打包的 OpenCV，不下载额外神经网络权重。HDR 使用 `createMergeMertens(1, 1, 1)`，配合曝光匹配和基于参考帧的运动区域替换；景深合成使用多尺度拉普拉斯清晰度与边界羽化；全景使用 SIFT、OpenCV detail 相机估计与优化、自行实现的分块球面投影和曝光补偿。依赖许可沿用 `THIRD_PARTY.md` 及 OpenCV 许可文件。

这些是 CPU 图像算法，不能称为 Adobe HDR / AI 景深算法。本版保留 1.2 的九个 ONNX 模型及其校验值，未更改 AI 超分和去杂色模型。

## 1.2 高画质增强

新增模型记录在 `assets/models/v12-models.json`，全部来源于上游官方权重。转换代码见 `tools/restoration_arch.py` 与 `tools/convert_restoration.py`；原始权重先验证 SHA-256，以 `torch.load(weights_only=True)` 读取并严格匹配网络。PyTorch / ONNX 仅开发转换使用，不打包进运行版。

| 用途 | 模型 | 参数数量 | 许可 |
|---|---|---:|---|
| 默认高画质超分 | Real-ESRGAN x4plus，23 RRDB | 16,697,987 | 权重 BSD-3-Clause；BasicSR 结构 Apache-2.0 |
| 默认高画质去杂色 | DRUNet color，带噪声条件的残差 U-Net | 32,640,960 | MIT，Kai Zhang |
| 可选真实噪声去杂色 | NAFNet SIDD width32 | 29,159,715 | MIT，Megvii |

官方权重：
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth
- NAFNet 官方 README 所列 Google Drive ID `1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ`，https://github.com/megvii-research/NAFNet

ONNX opset 17，动态尺寸。对 32×48、35×43 输入与原始 PyTorch 比较，RRDB 最大绝对误差 7.75e-7、DRUNet 3.58e-7、NAFNet 3.10e-6。DRUNet / NAFNet 内部动态补边后裁回输入尺寸。

RRDB 采用 128px 分块、32px 重叠和 32px 上下文；降噪为 256px 分块、64px 重叠、32px 上下文。重叠处余弦羽化并归一化，输出大于 256MiB 时使用磁盘映射。高画质局部预览只计算覆盖中央区域的块，网格、上下文和噪声参考均与整图一致。有限上下文仍不能保证与无限制整图神经推理完全相同。

DRUNet 使用原图九区 Haar 噪声估计，强度 35 为基准，sigma 上限 50/255。NAFNet 为 SIDD 真实照片噪声训练的固定模型，使用噪声参考调节结果与原图混合，默认强度 70。旧 FFDNet 和 compact 超分仍可选。算法并非对所有照片都更优；测试中 DRUNet 与 FFDNet 的合成噪声指标相近，NAFNet 在该样例残留噪点更多，详见测试报告。

所有模型处理显影后的 RGB；不等同 Adobe RAW 域增强。推理线程上限 32，实际取本机逻辑核心数与 32 的较小值；单一重图像任务运行，ONNX 跨算子线程 1。CPU 推理已验证，CUDA provider 接口保留但本轮没有 GPU 实测。

## 1.1 近景与去杂色

新增模型校验值见 `assets/models/v11-models.json`，均可离线运行：

- **MiDaS v2.1 small**：Intel ISL，MIT；官方 ONNX https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx 。256×256 RGB，ImageNet 归一化，输出相对逆深度（越大越近）。用 10%–95% 分位归一化，平滑阈值选近处，再用引导滤波对齐边缘。不是绝对距离测量。
- **FFDNet color**：KAIR / Kai Zhang，MIT；官方权重 https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth 。12 层卷积、96 个特征通道，PixelUnshuffle / PixelShuffle，RGB 和噪声标准差两个输入。`tools/convert_denoise.py` 使用 PyTorch 2.14.0 导出 opset 17 动态尺寸 ONNX，权重加载使用 `weights_only=True`，转换最大绝对误差约 1.01e-6。

去杂色使用 256px 分块与 32px 上下文、偶数坐标对齐；输入奇数宽高时复制补边。噪声参考取全图九个原始像素区块的 Haar 高频 MAD 估计，预览和整图使用同一参考，35 为基准强度。最终噪声标准差上限 75/255。该模型针对 RGB 高斯型噪声训练，真实复杂噪声和纹理需要人工检查，不是 RAW 马赛克域降噪。

主体复用 U2NetP 前景概率，背景使用其反选，无额外下载。1.1 包含六个 ONNX 文件，1.2 增加三个，共九个；PyTorch 仅用于转换，不进入运行包。


## 1.0 自动蒙版

三个选区模型随包离线运行，推理代码在 `lumen/selection.py`，文件 SHA-256 见 `assets/models/selection-models.json`。

| 用途 | 模型与许可 | 输入及输出 |
|---|---|---|
| 天空 | Sky-Segmentation-and-Post-processing / SkySeg，MIT；ONNX https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx | RGB 320×320，取首个天空概率输出 |
| 人物 | TorchVision DeepLabV3 MobileNetV3 Large，BSD-3-Clause，COCO_WITH_VOC_LABELS_V1 权重 | RGB 384×384，21 类 softmax 的 person 类（15） |
| 背景 | U2NetP 显著前景模型，Apache-2.0；ONNX https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx | RGB 320×320，显著前景概率取反 |

天空和 U2NetP 使用官方预处理方式：按最大输入强度归一化，再使用 ImageNet 均值／标准差；人物模型直接使用 ImageNet 归一化。概率图回到预览尺寸后使用引导滤波细化边缘，保存为 PNG 软蒙版，导出时映射到全尺寸。AI 选区可人工增减，不保证发丝和半透明边缘完全准确。

人物权重来自 https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth 。用 `tools/convert_person.py` 转换为 opset 17 固定尺寸 ONNX。模型转换需要 PyTorch 2.14.0、TorchVision 0.29.0、ONNX 1.23.0 和 ONNX Runtime 1.30.0，仅开发者转换时需要，应用运行不包含 PyTorch。

示例：`python tools/convert_person.py C:\Models\person-deeplab.onnx`。脚本从 TorchVision 官方源下载权重，并比较 PyTorch / ONNX 概率输出。背景是显著前景的反选，不是通用实例分割；多人图的人物蒙版包含所有识别到的人，不逐个人物编号。

## 超分

运行 LUMEN RAW 不需要安装 PyTorch，也不需要另外下载权重。

随包提供的 `assets/models/realesr-general-x4v3.onnx` 来自 Real-ESRGAN 官方 v0.2.5.0 发布的 `realesr-general-x4v3.pth`，使用 BSD-3-Clause 许可。网络结构为 BasicSR SRVGGNetCompact：64 个特征通道、32 个中间卷积、PReLU、4× PixelShuffle，加最近邻放大的残差。

- 官方权重：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
- 原权重 SHA-256：`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`
- ONNX SHA-256：`abd38ff81ca7cd9e03e213394267c50d393c8021b64f2f57b818f1ec82ad2369`
- 转换：PyTorch 2.14.0 的 legacy ONNX exporter，opset 17，动态宽高，RGB NCHW float32 输入，范围 0–1，原生 4× 输出。
- 对同一固定输入比较原始 PyTorch 模型与 ONNX CPU 输出，最大绝对误差 `8.9407e-7`。
- 2× 模式在每个 4× 输出块上做面积降采样；分块输入为 192px，上下文为 40px。自动测试对比整图和分块真实推理，误差小于 `2e-5`。

网络生成的高频纹理不一定是真实细节。软件的增强处理发生在 RAW 显影后的 RGB 阶段，不能代替 RAW 传感器级细节重建，也不是 Adobe 增强算法。

开发者可使用 `tools/convert_model.py` 重新转换。先在单独环境安装 `torch==2.14.0`、`onnx==1.23.0`、`onnxruntime==1.30.0`，从上述官方地址下载权重，再运行：

```powershell
python tools/convert_model.py C:\Models\realesr-general-x4v3.pth C:\Models\converted.onnx
```

转换脚本验证输入权重的 SHA-256，并使用 `torch.load(weights_only=True)`。此脚本不是应用运行时的一部分。Real-ESRGAN、BasicSR 的许可证已放在 `assets` 目录中。
