"""1.3.0: one version number drives the window, installer and portable package."""
import re
from pathlib import Path
from lumen import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_single_sourced():
    assert re.fullmatch(r'\d+\.\d+\.\d+', __version__)
    iss = (ROOT / 'installer.iss').read_text(encoding='utf-8')
    assert f'#define AppVersion "{__version__}"' in iss
    assert '{#AppVersion}' in iss and 'AppVersion=1.' not in iss
    # Same AppId as 1.0 – 1.2.2 so the installer upgrades the existing installation in place.
    assert 'AppId={{F1CA8FF7-EB54-4B53-81E3-4CC183C8C1B9}' in iss
    package = (ROOT / 'tools' / 'package_windows.py').read_text(encoding='utf-8')
    assert 'from lumen import __version__' in package and '1.2.2' not in package
    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    assert f'## {__version__}' in changelog


def test_installer_replaces_previous_runtime_files():
    iss = (ROOT / 'installer.iss').read_text(encoding='utf-8')
    assert 'Type: filesandordirs; Name: "{app}\\_internal"' in iss
