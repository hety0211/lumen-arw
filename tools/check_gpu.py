"""Return nonzero unless every packaged pixel graph really executes GPU nodes."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen import compute, gpu_graphs, model

adapter = compute.preferred_adapter()
if adapter is None:
    raise SystemExit('No hardware DXGI adapter found.')
edits = model.recipe()
inputs = dict(gpu_graphs.tonal_inputs(edits['adjustments']), **gpu_graphs.color_inputs(edits))
image = np.full((1, 16, 16, 3), .25, np.float32)
for kind in ('tonal', 'color', 'fused'):
    session = compute.session(gpu_graphs.model(kind), True)
    names = {i.name for i in session.get_inputs()}
    result = session.run(None, {k: v for k, v in dict(inputs, image=image).items() if k in names})[0]
    provider = session.get_providers()[0]
    if provider == 'CPUExecutionProvider' or result.shape != image.shape or not np.isfinite(result).all():
        raise SystemExit(f'GPU inference verification failed for {kind}: ' + str(compute.state.snapshot()))
    print(f'{kind}: {provider} · {adapter[1]} · GPU node execution verified')

# Informational: what the Windows ML execution-provider catalog offers on this PC (1.3.1).
from lumen import winml
if not winml.supported():
    print(f'Windows ML catalog: not used on Windows build {winml.windows_build()} (needs {winml.MIN_BUILD}+)')
else:
    try:
        from windowsml import EpCatalog
        with EpCatalog() as catalog:
            offered = [f'{p.name} {p.version} ({p.ready_state.name})' for p in catalog.find_all_providers()]
        print('Windows ML catalog: ' + (', '.join(offered) or 'no execution providers offered for this hardware'))
    except Exception as exc:
        print(f'Windows ML catalog unavailable: {type(exc).__name__}: {exc}')
