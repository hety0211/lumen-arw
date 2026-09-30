# Windows x64 build; collect the active ONNX Runtime's DirectML DLL when present.
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
raw_data, raw_binaries, raw_hidden = collect_all('rawpy')
ort_data, ort_binaries, ort_hidden = collect_all('onnxruntime')
a = Analysis(
    [str(root / 'main.py')], pathex=[str(root)],
    binaries=raw_binaries + ort_binaries,
    datas=[(str(root / 'assets'), 'assets')] + raw_data + ort_data + collect_data_files('tifffile'),
    hiddenimports=raw_hidden + ort_hidden + ['PIL.ImageCms'],
    excludes=['cupy', 'torch', 'torchvision', 'onnx', 'sympy'], noarchive=False,
)
if os.name == 'nt':
    # Qt 6.11 uses the Windows ICU shim. A different ICU on PATH must not shadow
    # System32/icuuc.dll: it exports versioned symbols and prevents Qt loading.
    a.binaries = [item for item in a.binaries
                  if Path(item[0]).name.lower() != 'icuuc.dll'
                  and not Path(item[0]).name.lower().startswith('icudt')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LumenARW',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(root/'assets/lumen.ico'))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='LumenARW')
