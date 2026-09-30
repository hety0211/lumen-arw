"""Package the verified Windows onedir build as a portable ZIP."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'dist' / 'LumenRAW'
OUTPUT = PROJECT / '.publish' / 'v122' / 'packages' / 'LumenRAW-1.2.2-Windows.zip'


def main():
    if not (SOURCE / 'LumenRAW.exe').is_file():
        raise SystemExit('Missing frozen editor executable.')
    files = sorted(path for path in SOURCE.rglob('*') if path.is_file())
    names = {path.name.lower() for path in files}
    required = {'directml.dll', 'lumen.ico', 'notosanssc.ttf', 'exiftool.exe',
                'realesrgan-x4plus.onnx', 'realesr-general-x4v3.onnx',
                'nafnet-sidd.onnx', 'drunet-color.onnx', 'ffdnet-color.onnx',
                'skyseg.onnx', 'person-deeplab.onnx', 'u2netp.onnx', 'midas-small.onnx'}
    missing = required - names
    if missing:
        raise SystemExit('Frozen app is missing runtime resources: ' + ', '.join(sorted(missing)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(f'Preserving existing package: {OUTPUT}')
    temporary = OUTPUT.with_suffix('.zip.tmp')
    if temporary.exists():
        raise SystemExit(f'Remove the interrupted temporary archive after inspecting it: {temporary}')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=4, allowZip64=True) as archive:
            for path in files:
                archive.write(path, 'LumenRAW-Windows/' + path.relative_to(SOURCE).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise RuntimeError(f'Portable ZIP failed CRC verification: {corrupt}')
        temporary.replace(OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with OUTPUT.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'version': '1.2.2', 'archive': OUTPUT.name, 'sha256': digest,
              'bytes': OUTPUT.stat().st_size, 'files': len(files),
              'zip_crc_verified': True, 'distribution': 'Windows DirectML'}
    OUTPUT.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
