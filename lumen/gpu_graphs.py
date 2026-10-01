"""Pointwise development graphs for GPU execution providers (DirectML / CUDA).

Every graph takes and returns channel-last float32 tiles (1, H, W, 3), so the
NumPy image can be uploaded without a transpose.  Three graphs are built:

* ``tonal``: white-balance gains, exposure and the four tonal zones (1.2.2).
* ``color``: saturation / vibrance, 8-color HSL, RGB + channel curves,
  monochrome and three-way grading.
* ``fused``: ``tonal`` followed by ``color`` in one upload / download, used
  when no spatial detail tool (clarity, dehaze, ...) sits between them.

The CPU path in ``engine`` remains the reference implementation; the parity
tests compare both across non-default parameters.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .onnx_graph import Graph, INT64
from .model import COLORS

LUMA = np.array([.2126, .7152, .0722], np.float32).reshape(1, 1, 1, 3)
HUE_SAMPLES = 361          # 1 degree grid; every HSL center is an integer degree.
CURVE_SAMPLES = 4097       # identical axis to engine.apply_curves smooth mode.
EPS = float(np.finfo(np.float32).eps)
SHAPE = [1, 'height', 'width', 3]


def _scalar(g, value):
    return g.const(np.float32(value))


def _clip(g, x, low=0., high=1.):
    return g.op('Clip', x, _scalar(g, low), _scalar(g, high))


def _luma(g, x):
    return g.op('ReduceSum', g.op('Mul', x, LUMA), g.const([3], np.int64), keepdims=1)


def _channel(g, x, index):
    i64 = lambda v: g.const([v], np.int64)
    return g.op('Slice', x, i64(index), i64(index + 1), i64(3))


def _lookup(g, x, table, samples, low, high):
    """Linear interpolation into a 1-D table, matching numpy.interp on a grid."""
    position = g.op('Mul', _clip(g, x, low, high), _scalar(g, (samples - 1) / (high - low)))
    if low:
        position = g.op('Sub', position, _scalar(g, low * (samples - 1) / (high - low)))
    index = _clip(g, g.op('Floor', position), 0, samples - 2)
    fraction = g.op('Sub', position, index)
    index = g.op('Cast', index, to=INT64)
    left = g.op('Gather', table, index, axis=0)
    right = g.op('Gather', table, g.op('Add', index, g.const(1, np.int64)), axis=0)
    return g.op('Add', left, g.op('Mul', g.op('Sub', right, left), fraction))


def _tonal(g, x):
    gains = g.input('gains', [1, 1, 1, 3])
    exposure = g.input('exposure', [1])
    zones = {name: g.input(name, [1]) for name in ('shadows', 'highlights', 'blacks', 'whites')}
    contrast = g.input('contrast', [1])
    exposed = g.op('Mul', g.op('Mul', x, gains), exposure)
    p = g.op('Pow', _clip(g, _luma(g, exposed)), _scalar(g, .45))
    inv = g.op('Sub', _scalar(g, 1), p)
    middle = g.op('Add', g.op('Mul', zones['shadows'], g.op('Pow', inv, _scalar(g, 2))),
                  g.op('Mul', zones['highlights'], g.op('Pow', p, _scalar(g, 3))))
    ends = g.op('Add', g.op('Mul', zones['blacks'], g.op('Pow', inv, _scalar(g, 6))),
                g.op('Mul', zones['whites'], g.op('Pow', p, _scalar(g, 6))))
    stops = g.op('Add', g.op('Div', middle, _scalar(g, 65)), g.op('Div', ends, _scalar(g, 85)))
    linear = g.op('Max', g.op('Mul', exposed, g.op('Pow', _scalar(g, 2), stops)), _scalar(g, 0))
    srgb = g.op('Where', g.op('LessOrEqual', linear, _scalar(g, .0031308)),
                g.op('Mul', linear, _scalar(g, 12.92)),
                g.op('Sub', g.op('Mul', g.op('Pow', linear, _scalar(g, 1 / 2.4)), _scalar(g, 1.055)),
                     _scalar(g, .055)))
    centered = g.op('Sub', srgb, _scalar(g, .5))
    return _clip(g, g.op('Add', g.op('Mul', centered, contrast), _scalar(g, .5)))


def _hsl(g, x):
    tables = [g.input(name, [HUE_SAMPLES]) for name in ('hue_lut', 'sat_lut', 'val_lut')]
    r, gr, b = (_channel(g, x, i) for i in range(3))
    v = g.op('Max', r, gr, b)
    diff = g.op('Sub', v, g.op('Min', r, gr, b))
    s = g.op('Div', diff, g.op('Add', g.op('Abs', v), _scalar(g, EPS)))
    k = g.op('Div', _scalar(g, 60), g.op('Add', diff, _scalar(g, EPS)))
    h_r = g.op('Mul', g.op('Sub', gr, b), k)
    h_g = g.op('Add', g.op('Mul', g.op('Sub', b, r), k), _scalar(g, 120))
    h_b = g.op('Add', g.op('Mul', g.op('Sub', r, gr), k), _scalar(g, 240))
    h = g.op('Where', g.op('Equal', v, r), h_r, g.op('Where', g.op('Equal', v, gr), h_g, h_b))
    h = g.op('Where', g.op('Less', h, _scalar(g, 0)), g.op('Add', h, _scalar(g, 360)), h)
    dh, ds, dv = (_lookup(g, h, t, HUE_SAMPLES, 0., 360.) for t in tables)
    hue = g.op('Add', h, dh)
    hue = g.op('Sub', hue, g.op('Mul', g.op('Floor', g.op('Div', hue, _scalar(g, 360))), _scalar(g, 360)))
    s = _clip(g, g.op('Mul', s, g.op('Add', ds, _scalar(g, 1))))
    v = _clip(g, g.op('Mul', v, g.op('Pow', _scalar(g, 2), dv)))
    sector = g.op('Div', hue, _scalar(g, 60))
    vs = g.op('Mul', v, s)
    channels = []
    for n in (5., 3., 1.):
        t = g.op('Add', sector, _scalar(g, n))
        t = g.op('Sub', t, g.op('Mul', g.op('Floor', g.op('Div', t, _scalar(g, 6))), _scalar(g, 6)))
        weight = _clip(g, g.op('Min', t, g.op('Sub', _scalar(g, 4), t)))
        channels.append(g.op('Sub', v, g.op('Mul', vs, weight)))
    return g.op('Concat', *channels, axis=3)


def _color(g, x):
    saturation, vibrance = g.input('saturation', [1]), g.input('vibrance', [1])
    lum = _luma(g, x)
    spread = g.op('Sub', g.op('ReduceMax', x, axes=[3], keepdims=1), g.op('ReduceMin', x, axes=[3], keepdims=1))
    scale = g.op('Add', saturation, g.op('Mul', vibrance, g.op('Sub', _scalar(g, 1), spread)))
    x = _clip(g, g.op('Add', lum, g.op('Mul', g.op('Sub', x, lum), scale)))
    x = _hsl(g, x)
    curve = g.input('curve_rgb', [CURVE_SAMPLES])
    x = _lookup(g, x, curve, CURVE_SAMPLES, 0., 1.)
    tables = [g.input(f'curve_{c}', [CURVE_SAMPLES]) for c in 'rgb']
    x = g.op('Concat', *[_lookup(g, _channel(g, x, i), tables[i], CURVE_SAMPLES, 0., 1.) for i in range(3)], axis=3)
    mono = g.input('mono', [1])
    x = g.op('Add', x, g.op('Mul', mono, g.op('Sub', _luma(g, x), x)))
    balance = g.input('balance', [1])
    zones = [g.input(f'grade_{z}', [1, 1, 1, 3]) for z in ('shadows', 'midtones', 'highlights')]
    lum = _luma(g, x)
    tone = _clip(g, g.op('Add', lum, balance))
    inv = g.op('Sub', _scalar(g, 1), tone)
    weights = [g.op('Mul', inv, inv), g.op('Mul', g.op('Mul', tone, inv), _scalar(g, 2)), g.op('Mul', tone, tone)]
    protection = g.op('Add', _scalar(g, .2), g.op('Mul', _scalar(g, .8),
                      g.op('Sin', g.op('Mul', _clip(g, lum), _scalar(g, np.pi)))))
    tint = g.op('Add', g.op('Add', g.op('Mul', weights[0], zones[0]), g.op('Mul', weights[1], zones[1])),
                g.op('Mul', weights[2], zones[2]))
    return _clip(g, g.op('Add', x, g.op('Mul', protection, tint)))


@lru_cache(maxsize=None)
def model(kind):
    if kind not in ('tonal', 'color', 'fused'):
        raise ValueError(kind)
    g = Graph(f'LUMEN RAW {kind} pipeline')
    x = g.input('image', SHAPE)
    if kind in ('tonal', 'fused'):
        x = _tonal(g, x)
    if kind in ('color', 'fused'):
        x = _color(g, x)
    g.output(x, 'output', SHAPE)
    return g.serialize()


def tonal_inputs(a):
    temp, tint = a['temperature'] / 100, a['tint'] / 100
    gains = np.array([2 ** (.4 * temp + .15 * tint), 2 ** (-.15 * tint),
                      2 ** (-.4 * temp + .15 * tint)], np.float32).reshape(1, 1, 1, 3)
    scalar = lambda number: np.array([number], np.float32)
    return dict(gains=gains, exposure=scalar(2 ** a['exposure']),
                shadows=scalar(a['shadows']), highlights=scalar(a['highlights']),
                blacks=scalar(a['blacks']), whites=scalar(a['whites']),
                contrast=scalar(1 + a['contrast'] / 125))


def color_inputs(edits):
    import colorsys
    from . import curves as tone_curves
    a = edits['adjustments']
    scalar = lambda number: np.array([number], np.float32)
    hue_axis = np.arange(HUE_SAMPLES, dtype=np.float64)
    centers = [c[1] for c in COLORS] + [360]
    controls = np.asarray(list(edits['hsl']) + [edits['hsl'][0]], dtype=np.float64)
    inputs = dict(saturation=scalar(1 + a['saturation'] / 100), vibrance=scalar(a['vibrance'] / 100),
                  hue_lut=np.interp(hue_axis, centers, controls[:, 0] * .45).astype(np.float32),
                  sat_lut=np.interp(hue_axis, centers, controls[:, 1] / 100).astype(np.float32),
                  val_lut=np.interp(hue_axis, centers, controls[:, 2] / 100).astype(np.float32))
    axis = np.linspace(0, 1, CURVE_SAMPLES)
    mode = edits.get('curve_mode', 'linear')
    for channel, name in (('RGB', 'curve_rgb'), ('R', 'curve_r'), ('G', 'curve_g'), ('B', 'curve_b')):
        points = edits['curves'][channel]
        if channel == 'RGB':
            values = tone_curves.rgb_table(edits, axis)  # with the exposure curve's fine tone curve (1.4.1)
        else:
            values = axis if points == [[0., 0.], [1., 1.]] else tone_curves.evaluate(points, axis, mode)
        inputs[name] = np.asarray(values, np.float32)
    inputs['mono'] = scalar(1. if edits.get('monochrome', False) else 0.)
    grading = edits.get('grading', {})
    inputs['balance'] = scalar(grading.get('balance', 0) / 300)
    for zone in ('shadows', 'midtones', 'highlights'):
        hue, strength = grading.get(zone, [0, 0])
        color = np.asarray(colorsys.hsv_to_rgb((hue % 360) / 360, 1, 1), np.float32)
        color -= np.dot(color, [.2126, .7152, .0722])
        inputs[f'grade_{zone}'] = (color * np.float32(strength / 350)).astype(np.float32).reshape(1, 1, 1, 3)
    return inputs
