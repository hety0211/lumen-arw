"""Compare every DXGI GPU with the CPU on the 1.3.0 pixel pipeline.

Usage: python tools/bench_devices.py [output.json] [photo]
Each GPU is forced with LUMEN_DML_DEVICE, e.g. to model an integrated-GPU-only
machine on a desktop that also has a discrete card.  Reports, per device and for
a preview and a full-resolution image:
  * graph level: tonal / color / fused pixel graphs (GPU) vs NumPy (CPU);
  * pipeline level: uncached render, cached HSL change, cached mask change,
    with and without spatial detail tools.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen import compute, engine, gpu_graphs, performance  # noqa: E402
from tools.bench_pipeline import edits  # noqa: E402

SAMPLES = ('F:\\DCIM\\100MSDCF\\DSC07998.ARW', 'F:\\DCIM\\100MSDCF\\DSC08026.ARW')


def timed(fn, repeat):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return round(min(times), 1)


def synthetic(height, width):
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    sky = np.stack([.25 + .1 * y / height, .35 + .1 * y / height, .6 - .2 * y / height], 2)
    image = sky * (.6 + .4 * np.sin(x / 90)[..., None]) + rng.normal(0, .01, sky.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32)


def images(photo):
    if photo and Path(photo).exists():
        preview = engine.load_image(photo)[0]
        full = engine.load_image(photo, preview_limit=None)[0]
        return Path(photo).name, [('preview', preview, .4), ('full', full, 1.)]
    return 'synthetic', [('preview', synthetic(1067, 1600), .4), ('full', synthetic(4000, 6000), 1.)]


def graph_level(backend, image, e, repeat):
    a = e['adjustments']
    tonal = backend.tonal(image, a)
    if backend.gpu_pointwise:
        runs = {'tonal': lambda: backend._run_graph('tonal', image, gpu_graphs.tonal_inputs(a)),
                'color': lambda: backend._run_graph('color', tonal, gpu_graphs.color_inputs(e)),
                'fused': lambda: backend._run_graph('fused', image, dict(gpu_graphs.tonal_inputs(a), **gpu_graphs.color_inputs(e)))}
    else:
        runs = {'tonal': lambda: backend.tonal(image, a),
                'color': lambda: engine.color_stage(tonal, e),
                'fused': lambda: engine.color_stage(backend.tonal(image, a), e)}
    result = {}
    for name, fn in runs.items():
        start = time.perf_counter()
        fn()  # first call: session creation and GPU-node verification
        result[name + '_first_ms'] = round((time.perf_counter() - start) * 1000, 1)
        result[name + '_ms'] = timed(fn, repeat)
    return result


def pipeline_level(backend, image, scale, spatial, repeat, budget):
    e = edits(spatial)
    result = dict(uncached_ms=timed(lambda: engine.process(image, e, backend, apply_crop=False, detail_scale=scale), repeat))
    cache = engine.RenderCache(max_bytes=budget)
    engine.process(image, e, backend, apply_crop=False, detail_scale=scale, cache=cache)
    counter = iter(range(1, 1000))

    def change(kind):
        changed = copy.deepcopy(e)
        if kind == 'hsl':
            changed['hsl'][2][1] = float(next(counter) % 40)
        else:
            changed['masks'][0]['adjustments']['exposure'] = (next(counter) % 10) / 10
        engine.process(image, changed, backend, apply_crop=False, detail_scale=scale, cache=cache)
    result['hsl_change_ms'] = timed(lambda: change('hsl'), repeat)
    result['mask_change_ms'] = timed(lambda: change('mask'), repeat)
    return result


def device_report(label, backend, cases):
    report = dict(device=label, backend=backend.name, gpu_pointwise=backend.gpu_pointwise, sizes={})
    for size, image, scale in cases:
        repeat = 3 if size == 'preview' else 2
        budget = 512 * 2**20 if size == 'preview' else 3 * 2**30
        entry = dict(shape=list(image.shape), graphs=graph_level(backend, image, edits(False), repeat))
        entry['pointwise'] = pipeline_level(backend, image, scale, False, repeat, budget)
        entry['spatial'] = pipeline_level(backend, image, scale, True, repeat, budget)
        entry['provider_after'] = compute.state.snapshot()[0]
        report['sizes'][size] = entry
        print(f'  {label} {size}: {json.dumps(entry, ensure_ascii=False)}', flush=True)
    report['warning'] = backend.warning
    return report


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'bench-devices.json'
    photo = sys.argv[2] if len(sys.argv) > 2 else next((p for p in SAMPLES if Path(p).exists()), None)
    source, cases = images(photo)
    adapters = compute.dxgi_adapters()
    print(f'sample: {source}; adapters: {adapters}; CPU threads: {performance.THREADS}', flush=True)
    results = dict(sample=source, cpu_threads=performance.THREADS,
                   adapters=[dict(index=a[0], name=a[1], dedicated_mb=round(a[2] / 2**20), vendor=hex(a[3])) for a in adapters],
                   devices=[])
    for index, name, *_ in adapters:
        os.environ['LUMEN_DML_DEVICE'] = str(index)
        try:
            backend = engine.Backend('auto')
            if name not in backend.name:
                raise RuntimeError(f'expected {name}, backend chose {backend.name}')
            results['devices'].append(device_report(f'GPU {index} · {name}', backend, cases))
        except Exception as exc:
            results['devices'].append(dict(device=f'GPU {index} · {name}', error=repr(exc)))
            print(f'  GPU {index} failed: {exc!r}', flush=True)
    os.environ.pop('LUMEN_DML_DEVICE', None)
    results['devices'].append(device_report(f'CPU · {performance.THREADS} threads', engine.Backend('cpu'), cases))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nms (min)            ' + ''.join(f'{d["device"][:22]:>24}' for d in results['devices']))
    rows = [('preview', 'graphs', 'tonal_ms'), ('preview', 'graphs', 'color_ms'), ('preview', 'graphs', 'fused_ms'),
            ('preview', 'pointwise', 'uncached_ms'), ('preview', 'pointwise', 'hsl_change_ms'),
            ('preview', 'spatial', 'uncached_ms'), ('preview', 'spatial', 'mask_change_ms'),
            ('full', 'graphs', 'fused_ms'), ('full', 'pointwise', 'uncached_ms'), ('full', 'pointwise', 'hsl_change_ms'),
            ('full', 'spatial', 'uncached_ms'), ('full', 'spatial', 'mask_change_ms')]
    for size, group, key in rows:
        cells = []
        for d in results['devices']:
            value = d.get('sizes', {}).get(size, {}).get(group, {}).get(key)
            cells.append(f'{value if value is not None else "-":>24}')
        print(f'{size[:4]} {group[:5]} {key[:-3]:<14}' + ''.join(cells))
    print(f'\nSaved {output}')


if __name__ == '__main__':
    main()
