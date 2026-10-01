"""Install the pure-Perl Image-ExifTool distribution for macOS (1.3.1).

Windows runs exiftool.exe from the runtime assets.  Its Strawberry Perl library
cannot run on macOS, so the macOS build uses the same ExifTool version as plain
Perl modules with the system ``/usr/bin/perl``, installed to
``assets/exiftool/unix``.  The archive is verified against the SHA-256 published
on https://exiftool.org/checksums.txt before anything is extracted.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = '13.59'
SHA256 = '668ea3acececb7235fbd0f4900e72d5f12c9b07e5c778fd36cb1e9b5828fd65a'
URLS = (f'https://exiftool.org/Image-ExifTool-{VERSION}.tar.gz',
        f'https://sourceforge.net/projects/exiftool/files/Image-ExifTool-{VERSION}.tar.gz/download')
TARGET = ROOT / 'assets' / 'exiftool' / 'unix'
KEEP = ('exiftool', 'README', 'lib/')


def installed_version(target=TARGET):
    perl = '/usr/bin/perl' if Path('/usr/bin/perl').exists() else shutil.which('perl')
    if not perl or not (target / 'exiftool').is_file():
        return None
    result = subprocess.run([perl, str(target / 'exiftool'), '-ver'], capture_output=True, text=True, timeout=60)
    return result.stdout.strip() if result.returncode == 0 else None


def download():
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'LUMEN-RAW-build'})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except OSError as error:
            print(f'  {url}: {error}')
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest == SHA256:
            return data
        print(f'  {url}: SHA-256 {digest} does not match the published value')
    raise SystemExit('ExifTool download failed or did not match the published SHA-256.')


def extract(data, target=TARGET):
    prefix = f'Image-ExifTool-{VERSION}/'
    with tempfile.TemporaryDirectory(prefix='exiftool-') as temporary:
        staging = Path(temporary) / 'unix'
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if not member.name.startswith(prefix) or '..' in name.parts or name.is_absolute():
                    continue
                relative = member.name[len(prefix):]
                wanted = relative in KEEP or any(relative.startswith(k) for k in KEEP if k.endswith('/'))
                if not member.isfile() or not wanted:
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('wb') as stream:
                    shutil.copyfileobj(source, stream)
                destination.chmod(0o755 if relative == 'exiftool' else 0o644)
        if not (staging / 'exiftool').is_file() or not (staging / 'lib' / 'Image' / 'ExifTool.pm').is_file():
            raise SystemExit('Unexpected ExifTool archive layout.')
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, target)


def main():
    if installed_version() == VERSION:
        print(f'ExifTool {VERSION} (Perl) already installed in {TARGET.relative_to(ROOT)}')
        return 0
    print(f'Downloading Image-ExifTool {VERSION}')
    extract(download())
    version = installed_version()
    if version is not None and version != VERSION:
        raise SystemExit(f'Installed ExifTool reports version {version}, expected {VERSION}.')
    print(f'Installed ExifTool {VERSION} (Perl) in {TARGET.relative_to(ROOT)}'
          + ('' if version else ' (not run: no Perl on this machine)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
