"""Preview-pipeline timings for the release report.

Usage: python tools/bench_pipeline.py [photo] [output.json]
Measures a full uncached render, then cached re-renders after an HSL change and a
mask change, on the automatic backend (DirectML when available) and on the CPU.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen import compute, engine, model  # noqa: E402


def sample(path):
    if path:
        return engine.load_image(path)[0]
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:1067, 0:1600].astype(np.float32)
    sky = np.stack([.25 + .1 * y / 1067, .35 + .1 * y / 1067, .6 - .2 * y / 1067], 2)
    return np.ascontiguousarray(np.clip(sky * (.6 + .4 * np.sin(x / 90)[..., None]) + rng.normal(0, .01, sky.shape), 0, 1).astype(np.float32))


def edits(spatial):
    e = model.recipe()
    e['adjustments'].update(exposure=.3, shadows=30, highlights=-25, saturation=8, vibrance=12)
    if spatial:
        e['adjustments'].update(clarity=20, texture=15, sharpness=25)
    e['hsl'][5] = [10., -20., 5.]
    e['curves']['RGB'] = [[0, 0], [.35, .3], [.75, .8], [1, 1]]
    e['curve_mode'] = 'smooth'
    e['grading']['highlights'] = [45., 20.]
    for i, (start, end) in enumerate((([.05, .05], [.35, .3]), ([.6, .55], [.85, .9]))):
        m = model.new_mask('radial', i + 1)
        m.update(start=start, end=end)
        m['adjustments'].update(exposure=.4, clarity=15 if spatial else 0)
        e['masks'].append(m)
    return e


def best(fn, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return round(min(times), 1)


def measure(source, backend, spatial):
    e = edits(spatial)
    uncached = best(lambda: engine.process(source, e, backend, apply_crop=False, detail_scale=.4))
    cache = engine.RenderCache()
    engine.process(source, e, backend, apply_crop=False, detail_scale=.4, cache=cache)
    # Start at 1 so the first HSL change differs from the cached value (0).
    counter = iter(range(1, 1000))

    def hsl():
        changed = copy.deepcopy(e)
        changed['hsl'][2][1] = float(next(counter) % 40)
        engine.process(source, changed, backend, apply_crop=False, detail_scale=.4, cache=cache)

    def mask():
        changed = copy.deepcopy(e)
        changed['masks'][0]['adjustments']['exposure'] = (next(counter) % 10) / 10
        engine.process(source, changed, backend, apply_crop=False, detail_scale=.4, cache=cache)
    return dict(uncached_ms=uncached, hsl_change_ms=best(hsl), mask_change_ms=best(mask))


def main():
    source = sample(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None)
    report = dict(shape=list(source.shape))
    for label, backend in (('auto', engine.Backend('auto')), ('cpu', engine.Backend('cpu'))):
        report[label] = dict(backend=backend.name, spatial=measure(source, backend, True),
                             pointwise=measure(source, backend, False), gpu_pointwise=backend.gpu_pointwise)
    report['compute'] = dict(zip(('provider', 'device', 'detail', 'warning'), compute.state.snapshot()))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
