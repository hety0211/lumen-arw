import hashlib
import importlib.util
from pathlib import Path
import zipfile
import pytest

spec=importlib.util.spec_from_file_location('fetch_assets',Path(__file__).resolve().parents[1]/'tools/fetch_assets.py')
fetch=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetch)


def bundle(tmp_path, name='assets/models/example.onnx', content=b'test resource'):
    archive=tmp_path/'resources.zip'
    with zipfile.ZipFile(archive,'w') as stream:stream.writestr(name,content)
    manifest={'bundle':{'sha256':fetch.sha256(archive)},'files':[{'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}
    return archive,manifest


def test_asset_install_and_verify(tmp_path):
    archive,manifest=bundle(tmp_path)
    fetch.install_archive(archive,tmp_path,manifest)
    assert not fetch.verify(tmp_path,manifest)
    (tmp_path/'assets/models/example.onnx').write_bytes(b'changed')
    assert fetch.verify(tmp_path,manifest)==['assets/models/example.onnx']


def test_archive_corruption_preserves_existing_file(tmp_path):
    archive,manifest=bundle(tmp_path)
    existing=tmp_path/'assets/models/example.onnx';existing.parent.mkdir(parents=True);existing.write_bytes(b'keep')
    archive.write_bytes(archive.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='SHA-256'):fetch.install_archive(archive,tmp_path,manifest)
    assert existing.read_bytes()==b'keep'


@pytest.mark.parametrize('name', ['../escape.txt','assets/../../escape.txt','assets/C:/escape.txt','assets/..\\escape.txt'])
def test_archive_paths_cannot_escape_project(tmp_path,name):
    archive,manifest=bundle(tmp_path,name)
    with pytest.raises(ValueError,match='Invalid asset path'):fetch.install_archive(archive,tmp_path,manifest)


def test_member_hash_failure_changes_nothing(tmp_path):
    archive,manifest=bundle(tmp_path)
    manifest['files'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='Asset SHA-256'):fetch.install_archive(archive,tmp_path,manifest)
    assert not (tmp_path/'assets/models/example.onnx').exists()
