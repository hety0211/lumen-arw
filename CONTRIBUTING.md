# Contributing / 参与贡献

Bug reports, reproducible examples, documentation improvements, and focused pull requests are welcome. The application interface is currently Chinese; issues may be written in Chinese or English.

欢迎提交问题、可复现样例、文档改进和范围明确的 Pull Request。可以使用中文或英文交流。

## Local development

1. Use Windows and Python 3.12 x64; create a virtual environment.
2. Install `requirements-dev.txt` and run `python tools/fetch_assets.py`.
3. Run `python -m pytest tests -q`. The full suite includes actual local model inference and requires runtime assets.
4. Run `python main.py` to inspect changes in the desktop application.

Keep image processing off the UI thread. Preserve original photographs, avoid overwriting output names, and retain compatibility with existing `.lumen` / `.lumenalbum` recipes. Add meaningful regression coverage for processing, state, and I/O changes.

图像处理应放在后台线程，保留磁盘原片、避免覆盖同名副本，并注意旧工程兼容性。涉及处理结果、状态或文件读写的修改，请加入相应回归检查。

## Reporting a problem

Include the application version, Windows version, camera/model and RAW format, reproduction steps, and whether the issue occurs in the original-resolution view or exported image. Share a small sample only if you own it or have permission. Do not post passwords, API tokens, private paths, or unredacted sensitive EXIF information.

报告问题时请注明软件版本、Windows 版本、相机及 RAW 类型、复现步骤，以及问题出现在预览还是导出。样片需有分享权限，请先移除敏感信息。

## Repository contents

Do not commit model binaries, camera originals, personal edit projects, installers, virtual environments, or caches. Runtime assets are distributed in Releases and described by `assets/runtime-assets.json`; changing them requires updating sizes and SHA-256 checksums as well as model provenance and license notices.

Contributions are submitted under the project's MIT license. Third-party code or assets must retain compatible provenance and licensing information. Do not add official brand logos without redistribution rights.
