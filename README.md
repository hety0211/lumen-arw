# Lumen ARW

**开源多品牌 RAW 照片编辑器，集调色、AI 增强、智能蒙版与照片合成于一体。**

**An open-source multi-brand RAW photo editor with color grading, AI enhancement, smart masks, and photo merging.**

[中文使用说明](README.zh-CN.md) · [Downloads](https://github.com/hety0211/lumen-arw/releases) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

Lumen ARW is a local desktop photography workspace for Windows, designed around landscape and travel editing. The application currently has a **Chinese interface**. No account or cloud service is required to edit photographs; after the runtime assets are installed, editing and bundled AI inference work offline.

![Lumen ARW 1.2.1 desktop workspace](docs/screenshots/workspace.png)

Screenshot uses public CC0 test photographs; [image credits](docs/screenshots/README.md).

## Features

- **Multi-brand RAW:** Sony ARW, Canon CR2 / CR3, Nikon NEF, Fujifilm RAF, Panasonic RW2, DNG, and other formats supported by the bundled LibRaw version.
- **Non-destructive development:** exposure, highlights, shadows, whites, blacks, white balance, per-color HSL, grading, RGB channel curves, snapshots, and portable edit recipes.
- **Detail tools:** dehaze, clarity, texture, sharpening, noise reduction, spot healing, and clone stamping.
- **Masks:** brush and gradient masks, luminance ranges, contiguous color selection, and local models for sky, people, subject, background, and relative-depth foreground selections.
- **AI enhancement:** Real-ESRGAN super-resolution and DRUNet / NAFNet / FFDNet denoising in dedicated preview dialogs, creating editable DNG copies in the filmstrip.
- **Photo merging:** focus stacking, **Mertens exposure fusion** with three deghosting strengths, and spherical panoramas capped at **200 megapixels**.
- **Photo workflow:** original-resolution zoom, confirmed crop view, multi-select filmstrip, edit synchronization, and metadata watermark borders.
- **Export:** JPEG, PNG, 16-bit TIFF, and 16-bit linear RGB DNG. General processing/output limits are 400 megapixels, subject to available memory and disk space.

## Download and run

Get the Windows x64 installer or portable ZIP from [Releases](https://github.com/hety0211/lumen-arw/releases/latest). Windows 10 22H2 or Windows 11 is recommended. Portable builds include Python, ExifTool, fonts, and all nine ONNX models; extract the whole folder and keep `_internal` next to `LumenARW.exe`.

For existing installations, save your album and close the previous version before upgrading. Release binaries are currently unsigned.

## Run from source

Use **Python 3.12 x64** on Windows. Git contains application code, tests, model metadata, and license notices. Large runtime resources are versioned as a Release asset to keep the repository small.

```powershell
git clone https://github.com/hety0211/lumen-arw.git
cd lumen-arw
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/fetch_assets.py
.\.venv\Scripts\python.exe main.py
```

`fetch_assets.py` downloads the versioned runtime ZIP and verifies the archive and every extracted file against SHA-256 values in [runtime-assets.json](assets/runtime-assets.json). It restores the ONNX models, Noto font, ExifTool, and related notices. The initial download is approximately 600 MB; no model download is needed while editing afterward.

To install resources from an already downloaded Release asset:

```powershell
.\.venv\Scripts\python.exe tools/fetch_assets.py --archive C:\Downloads\LumenARW-1.2.1-RuntimeAssets.zip
.\.venv\Scripts\python.exe tools/fetch_assets.py --verify-only
```

After the initial resource setup, `run-source.cmd` can create/install the Python environment and start the application. See [MODEL.md](MODEL.md) for model origins, hashes, conversions, and limitations.

## Photo merging in 1.2.1

Select **2–32 photographs** with Ctrl / Shift in the filmstrip, then right-click → 合成 (Merge). Each method opens its own preview/progress dialog. The default reference is the image with the most original pixels; equal sizes select the middle frame. You can override the reference manually.

Results are named `reference-景深合成.dng`, `reference-HDR堆栈.dng`, or `reference-全景合成.dng`, with a numeric suffix on collision, and automatically added to the filmstrip. Deleting a filmstrip entry removes it from the album, **not from disk**.

HDR uses OpenCV Mertens exposure fusion, not radiance-map HDR. Focus stacking selects sharp regions at several scales. Panoramas use SIFT, camera optimization, spherical projection, exposure compensation, and tiled feather blending. Motion, parallax, focus breathing, and transparent edges can still produce artifacts.

## Acceleration and limits

Bundled Windows releases use **CPU inference**, with up to 32 compute threads. The source supports optional ONNX Runtime CUDA and CuPy; CUDA has **not been validated on NVIDIA hardware** in this release. See the [Chinese setup guide](README.zh-CN.md#超分与-cuda) and `requirements-gpu.txt` before installing GPU dependencies.

AI enhancement operates on developed RGB, not sensor mosaic data, and is not Adobe's RAW enhancement algorithm. Exported DNGs contain developed linear RGB pixels, not lossless copies of the original sensor mosaic. Keep your camera originals and edit recipes.

The 1.2.1 desktop release passed 166 regression checks; the open-source checkout passed **173 tests**, including seven resource-installer checks. A controlled large-array panorama test processed **199,197,856 pixels**. Photo merge workflows were tested with controlled views derived from one photograph; independently captured real-world brackets and panoramas still need validation. See [TEST_REPORT.md](TEST_REPORT.md) for scope and limitations.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/fetch_assets.py
.\.venv\Scripts\python.exe -m pytest tests -q
powershell -ExecutionPolicy Bypass -File build.ps1
```

`requirements-lock.txt` records the original Windows release environment. The PyInstaller specification builds a CPU-only portable application. To compile an installer with Inno Setup 7:

```powershell
ISCC.exe /DAppBuild=dist\LumenARW installer.iss
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and contributions. Please provide only photographs you have permission to share and remove sensitive location/person information before posting samples publicly.

## License and acknowledgments

Application code is released under the **[MIT License](LICENSE)**. Third-party dependencies, models, fonts, and ExifTool retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md), [MODEL.md](MODEL.md), and notices in `assets/`. The application does not bundle official camera/lens brand logos.

Built with Python, PySide6 / Qt, NumPy, OpenCV, rawpy / LibRaw, Pillow, tifffile, and ONNX Runtime, with models from Real-ESRGAN, KAIR / DPIR, NAFNet, MiDaS, U²-Net, SkySeg, and TorchVision.
