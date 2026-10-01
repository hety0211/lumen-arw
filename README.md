# LUMEN RAW

**开源多品牌 RAW 照片编辑器，集调色、AI 增强、智能蒙版与照片合成于一体。**

**An open-source multi-brand RAW photo editor with color grading, AI enhancement, smart masks, and photo merging.**

[中文使用说明](README.zh-CN.md) · [Downloads](https://github.com/hety0211/lumen-raw/releases) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

LUMEN RAW is a local desktop photography workspace for Windows and Apple silicon Macs, designed around landscape and travel editing. The application currently has a **Chinese interface**. No account or cloud service is required to edit photographs; after the runtime assets are installed, editing and bundled AI inference work offline.

![LUMEN RAW workspace shown under its former name in version 1.2.1](docs/screenshots/workspace.png)

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

Get the Windows x64 installer or portable ZIP from [Releases](https://github.com/hety0211/lumen-raw/releases/latest). Windows 10 22H2 or Windows 11 is recommended. Portable builds include Python, ExifTool, fonts, and all nine ONNX models; extract the whole folder and keep `_internal` next to `LumenRAW.exe`.

For existing installations, save your album and close the previous version before upgrading. Release binaries are currently unsigned.

### macOS (Apple silicon)

[`LumenRAW-1.3.1-macOS-arm64.dmg`](https://github.com/hety0211/lumen-raw/releases/tag/v1.3.1-macos) runs on M1 and newer Macs with macOS 15 Sequoia or later (the PySide6 6.11 bindings are built for macOS 15; every Apple silicon Mac can update to it). Features and the project / album formats match the Windows version.

- **Install:** open the DMG and drag **LUMEN RAW** to Applications. The app is ad-hoc signed, not notarized by Apple: the first launch is blocked; open System Settings → Privacy & Security and click **Open Anyway** (or run `xattr -dr com.apple.quarantine "/Applications/LUMEN RAW.app"`).
- **GPU:** the pointwise development stages that run on DirectML on Windows (white balance, exposure, tone zones, HSL, curves, grading, ...) run as **Metal** compute kernels; AI super-resolution, denoising and automatic masks run through ONNX Runtime's **Core ML** execution provider on the Apple GPU instead of Windows ML / TensorRT for RTX. Both are checked against the CPU on first use and fall back to it on failure. On an M1 Pro a 1600 px preview develops in about 3–5 ms (NumPy: ~760 ms) and the restoration models run 8–10x faster than on the CPU.
- **Mac conventions:** ⌘ shortcuts, Option-click to set the clone source, two-finger pan, pinch to zoom, Finder "Open With" and Dock drops. Logs are in `~/Library/Logs/LUMEN RAW`; ExifTool runs with the system Perl.
- **From source:** with Python 3.12, `./run-source.command` runs the editor and `./build-macos.command` runs the tests, the Metal / Core ML check, PyInstaller and the DMG build (output in `.publish/v131/macos/`).

### Version 1.3.1

- **NVIDIA RTX:** on Windows 11 24H2+ with a GeForce RTX 30-series or newer GPU, AI super-resolution, denoising and automatic masks use NVIDIA TensorRT for RTX, downloaded once through the Windows ML execution-provider catalog. Other PCs keep DirectML, then CPU.
- **Crash-safe AI:** neural models run in a separate worker process. If a GPU driver or execution provider crashes, the editor keeps running, retries the tile on the next device, and remembers the GPU + driver combination.
- ONNX Runtime is now the Windows ML build (`onnxruntime-windowsml` 1.30, CPU and DirectML built in).

### Version 1.3.0

- **Faster previews:** the develop pipeline caches each stage (repair/base, tone, spatial detail, color), so a slider only re-renders the stages after it. Local adjustments are computed only inside each mask's padded bounding box. Both are pixel-identical to a full re-render.
- **More work on the GPU:** on Windows, DirectML now runs saturation/vibrance, 8-color HSL, RGB and channel curves, monochrome and color grading in addition to tone. Without spatial detail tools, tone and color run as one GPU pass. The graphs are generated at runtime without the `onnx` package.
- **Explicit work state:** loading, export, AI/merge, mask inference, previews, detail rendering and thumbnails share one conflict table and one job scheduler.
- Diagnostic logs are written to `%LOCALAPPDATA%\LUMEN RAW\logs`. `LUMEN_COMPUTE=cpu` forces the CPU; CuPy is experimental and opt-in (`LUMEN_EXPERIMENTAL_CUPY=1`).

The installer keeps the original AppId, so it upgrades an existing 1.2.x installation in place. Project and album formats are unchanged.

### GPU acceleration

On Windows, LUMEN RAW uses AMD DirectML for pointwise development and the ONNX models. It picks the DXGI adapter with the most dedicated VRAM, checks that model nodes actually execute on GPU, and falls back to CPU when needed. The center of the bottom bar shows `GPU⚡` or `CPU⚡（thread count）`. RAW decoding and spatial tools (clarity, texture, dehaze, sharpening, noise reduction, retouching) still use CPU.

For a Python 3.12 x64 source environment, install the normal dependencies and runtime assets as below, then switch the ONNX Runtime distribution:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-directml
.\.venv\Scripts\python.exe -m pip install -r requirements-directml.txt
.\.venv\Scripts\python.exe main.py
```

The installer and portable ZIP are available from [Releases](https://github.com/hety0211/lumen-raw/releases/latest) and are rebuilt locally in `.publish/v131/packages/`, with SHA-256 values in `SHA256SUMS.txt`. They are unsigned. `build-directml.ps1` verifies actual GPU node execution before freezing the application; `installer.iss` is the Inno Setup recipe. NVIDIA CUDA still uses the separate `requirements-gpu.txt` flow and has not been validated here. See [TEST_REPORT.md](TEST_REPORT.md) for packaged ARW workflow checks.

## Run from source

Use **Python 3.12 x64** on Windows. Git contains application code, tests, model metadata, and license notices. Large runtime resources are versioned as a Release asset to keep the repository small.

```powershell
git clone https://github.com/hety0211/lumen-raw.git
cd lumen-raw
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

The Windows build uses DirectML on supported AMD GPUs for pointwise development (tone and color) and ONNX inference, with CPU fallback and up to 32 compute threads. The source supports optional ONNX Runtime CUDA and CuPy; CUDA has **not been validated on NVIDIA hardware** in this release. See the [Chinese setup guide](README.zh-CN.md#超分与-cuda) and `requirements-gpu.txt` before installing GPU dependencies.

AI enhancement operates on developed RGB, not sensor mosaic data, and is not Adobe's RAW enhancement algorithm. Exported DNGs contain developed linear RGB pixels, not lossless copies of the original sensor mosaic. Keep your camera originals and edit recipes.

Version 1.3.0 passed **193 regression tests** on the DirectML build (1.2.2: 176; 1.2.1: 166). A controlled large-array panorama test processed **199,197,856 pixels**. Photo merge workflows were tested with controlled views derived from one photograph; independently captured real-world brackets and panoramas still need validation. See [TEST_REPORT.md](TEST_REPORT.md) for scope and limitations.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/fetch_assets.py
.\.venv\Scripts\python.exe -m pytest tests -q
powershell -ExecutionPolicy Bypass -File build.ps1
```

`requirements-lock.txt` records the original Windows release environment. `build.ps1` builds a CPU portable app; `build-directml.ps1` builds the tested DirectML app. To compile an installer with Inno Setup 7:

```powershell
ISCC.exe /DAppBuild=dist\LumenRAW installer.iss
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and contributions. Please provide only photographs you have permission to share and remove sensitive location/person information before posting samples publicly.

## License and acknowledgments

Application code is released under the **[MIT License](LICENSE)**. Third-party dependencies, models, fonts, and ExifTool retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md), [MODEL.md](MODEL.md), and notices in `assets/`. The application does not bundle official camera/lens brand logos.

Built with Python, PySide6 / Qt, NumPy, OpenCV, rawpy / LibRaw, Pillow, tifffile, and ONNX Runtime, with models from Real-ESRGAN, KAIR / DPIR, NAFNet, MiDaS, U²-Net, SkySeg, and TorchVision.
