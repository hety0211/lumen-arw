import copy
import hashlib
import json
import sys
import types
import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image
from lumen import engine, model


@pytest.fixture
def gradient():
    value = np.linspace(.002, .55, 300, dtype=np.float32)
    return np.repeat(np.repeat(value[None, :, None], 80, axis=0), 3, axis=2)


def test_identity_and_non_destructive(gradient):
    original = gradient.copy()
    out = engine.process(gradient, model.recipe())
    np.testing.assert_allclose(out, engine.to_srgb(gradient), atol=1e-6)
    np.testing.assert_array_equal(gradient, original)
    np.testing.assert_allclose(engine.to_linear(out), gradient, atol=1e-6)


def test_one_ev_doubles_linear_values():
    src = np.full((8, 8, 3), .05, np.float32)
    edits = model.recipe()
    edits['adjustments']['exposure'] = 1
    np.testing.assert_allclose(engine.to_linear(engine.process(src, edits)), src * 2, atol=1e-6)


def test_shadow_and_highlight_selectivity(gradient):
    before = engine.process(gradient, model.recipe())
    e = model.recipe()
    e['adjustments']['shadows'] = 50
    shadow = engine.process(gradient, e) - before
    assert shadow[:, 10].mean() > shadow[:, -10].mean() * 2
    e = model.recipe()
    e['adjustments']['highlights'] = -50
    high = before - engine.process(gradient, e)
    assert high[:, -10].mean() > high[:, 10].mean() * 5


@pytest.mark.parametrize('key,value', [('dehaze', 70), ('dehaze', -70), ('clarity', 100),
    ('sharpness', 100), ('denoise', 100), ('color_noise', 100), ('temperature', -100), ('contrast', 100)])
def test_adjustments_finite_and_bounded(gradient, key, value):
    e = model.recipe()
    e['adjustments'][key] = value
    out = engine.process(gradient, e)
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert not np.allclose(out, engine.process(gradient, model.recipe()), atol=1e-7) or key == 'color_noise'


def test_denoise_reduces_noise():
    rng = np.random.default_rng(42)
    rgb = np.clip(.45 + rng.normal(0, .035, (160, 160, 3)), 0, 1).astype(np.float32)
    a = model.adjustments()
    a.update(denoise=100, color_noise=100)
    out = engine.details(rgb, a)
    assert np.var(out) < np.var(rgb) * .6


def test_color_region_selection():
    rgb = np.array([[[.8, .05, .05], [.05, .1, .8]]], np.float32)
    hsl = model.recipe()['hsl']
    hsl[0][1] = -100
    out = engine.apply_hsl(rgb, hsl)
    assert out[0, 0].max() - out[0, 0].min() < .05
    np.testing.assert_allclose(out[0, 1], rgb[0, 1], atol=1e-6)


def test_curve_rgb_channel():
    rgb = np.full((10, 10, 3), .25, np.float32)
    curves = model.recipe()['curves']
    curves['R'] = [[0., 0.], [.25, .5], [1., 1.]]
    out = engine.apply_curves(rgb, curves)
    np.testing.assert_allclose(out[..., 0], .5)
    np.testing.assert_allclose(out[..., 1:], .25)


def test_brush_localization_and_eraser():
    mask = model.new_mask('brush', 1)
    mask['strokes'] = [dict(points=[[.5, .5]], radius=.2, erase=False)]
    alpha = engine.mask_alpha(mask, (100, 100, 3))
    assert alpha[50, 50] == 1 and alpha[0, 0] == 0 and 0 < alpha[50, 66] < 1
    mask['strokes'].append(dict(points=[[.5, .5]], radius=.06, erase=True))
    erased = engine.mask_alpha(mask, (100, 100, 3))
    assert erased[50, 50] == 0 and erased[50, 60] > .9


def test_mask_invert_opacity_radial_linear():
    m = model.new_mask('radial', 1)
    a = engine.mask_alpha(m, (100, 100))
    assert a[50, 50] == 1 and a[0, 0] == 0
    m.update(invert=True, opacity=50)
    b = engine.mask_alpha(m, (100, 100))
    np.testing.assert_allclose(b, (1 - a) * .5)
    m = model.new_mask('linear', 1)
    m.update(start=[0., .5], end=[1., .5])
    a = engine.mask_alpha(m, (100, 100))
    assert a[50, 0] == 0 and a[50, -1] == 1
    assert np.all(np.diff(a[50]) >= 0)


def test_local_mask_changes_only_selection():
    src = np.full((100, 100, 3), .12, np.float32)
    e = model.recipe()
    m = model.new_mask('radial', 1)
    m['adjustments']['exposure'] = 1
    e['masks'].append(m)
    out = engine.process(src, e)
    before = engine.to_srgb(src)
    np.testing.assert_allclose(out[0, 0], before[0, 0], atol=1e-6)
    assert out[50, 50, 0] > before[50, 50, 0] * 1.3
    m['enabled'] = False
    np.testing.assert_allclose(engine.process(src, e), before, atol=1e-6)


def test_crop_rotation_matches_pixels():
    rgb = np.arange(6 * 8 * 3).reshape(6, 8, 3)
    e = model.recipe()
    e.update(crop=[.25, 0., .75, 1.], rotation=1)
    out = engine.crop_rotate(rgb, e)
    np.testing.assert_array_equal(out, np.rot90(rgb[:, 2:6], -1))


def test_tiff_16bit_icc_roundtrip(tmp_path, gradient):
    p = tmp_path / '测试.tif'
    rgb = engine.to_srgb(gradient)
    engine.export_image(p, rgb)
    with tifffile.TiffFile(p) as f:
        assert f.asarray().dtype == np.uint16
        assert 34675 in f.pages[0].tags
    loaded, info = engine.load_image(p, None)
    np.testing.assert_allclose(loaded, gradient, atol=3e-5)


@pytest.mark.parametrize('suffix', ['.jpg', '.png'])
def test_export_load_icc(tmp_path, gradient, suffix):
    p = tmp_path / ('图像' + suffix)
    engine.export_image(p, engine.to_srgb(gradient))
    with Image.open(p) as image:
        assert image.info['icc_profile']
    loaded, info = engine.load_image(p, 150)
    assert loaded.shape == (40, 150, 3)
    assert info['width'] == 300


def test_super_resolution_and_limit(monkeypatch):
    src = np.random.default_rng(2).random((20, 30, 3)).astype(np.float32)
    out, name = engine.super_resolve(src, 2)
    assert out.shape == (40, 60, 3) and np.isfinite(out).all()
    assert np.mean(np.abs(cv2.resize(out, (30, 20), interpolation=cv2.INTER_AREA) - src)) < .07
    monkeypatch.setattr(engine, 'MAX_EXPORT_PIXELS', 1000)
    with pytest.raises(ValueError):
        engine.super_resolve(src, 2)


def test_onnx_tiles_and_scale_validation(monkeypatch):
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def get_inputs(self):
            return [types.SimpleNamespace(name='input', type='tensor(float)', shape=[1, 3, 'h', 'w'])]
        def get_providers(self):
            return ['CPUExecutionProvider']
        def run(self, _, inputs):
            a = inputs['input']
            return [np.repeat(np.repeat(a, 2, axis=2), 2, axis=3)]
    monkeypatch.setitem(sys.modules, 'onnxruntime', types.SimpleNamespace(InferenceSession=Session, get_available_providers=lambda: ['CPUExecutionProvider']))
    rgb = np.random.default_rng(2).random((219, 270, 3)).astype(np.float32)
    out, _ = engine.onnx_super_resolve(rgb, 2, 'stub.onnx', True)
    np.testing.assert_array_equal(out, np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1))
    with pytest.raises(ValueError):
        engine.onnx_super_resolve(rgb, 4, 'stub.onnx', False)


def test_backend_runtime_fallback(gradient):
    backend = engine.Backend('cpu')
    backend.xp = types.SimpleNamespace()  # simulate a failing optional CUDA runtime
    out = backend.tonal(gradient, model.adjustments())
    assert backend.xp is np and 'CUDA' in backend.warning
    np.testing.assert_allclose(out, engine.to_srgb(gradient), atol=1e-6)


def test_project_history_roundtrip(tmp_path):
    e = model.recipe()
    e['masks'].append(model.new_mask('radial', 1))
    p, src = tmp_path / '编辑.lumen', tmp_path / '原片.ARW'
    src.write_bytes(b'original')
    digest = hashlib.sha256(src.read_bytes()).digest()
    model.save_project(p, src, e)
    source, loaded = model.load_project(p)
    assert loaded == e and PathLike(source) == PathLike(src)
    assert hashlib.sha256(src.read_bytes()).digest() == digest
    history = model.History(e)
    e['adjustments']['exposure'] = 1
    history.push(e)
    assert history.move(-1)['adjustments']['exposure'] == 0
    assert history.move(1)['adjustments']['exposure'] == 1


def PathLike(p):
    from pathlib import Path
    return Path(p).resolve()


def test_project_validation():
    e = model.recipe()
    e['adjustments']['exposure'] = float('nan')
    with pytest.raises(ValueError):
        model.validate(e)
    e = model.recipe()
    e['crop'] = [.8, 0, .2, 1]
    with pytest.raises(ValueError):
        model.validate(e)


def test_invalid_file(tmp_path):
    p = tmp_path / 'bad.arw'
    p.write_bytes(b'not a RAW')
    with pytest.raises(Exception):
        engine.load_image(p)
