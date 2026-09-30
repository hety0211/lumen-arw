"""Return nonzero unless the packaged tonal graph really executes GPU nodes."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen import compute
from lumen.tonal_dml import MODEL

adapter = compute.preferred_adapter()
if adapter is None:
    raise SystemExit('No hardware DXGI adapter found.')
session = compute.session(MODEL, True)
scalar = lambda value: np.array([value], np.float32)
inputs = dict(image=np.ones((1, 3, 16, 16), np.float32) * .25,
              gains=np.ones((1, 3, 1, 1), np.float32), exposure=scalar(1),
              shadows=scalar(0), highlights=scalar(0), blacks=scalar(0),
              whites=scalar(0), contrast=scalar(1))
result = session.run(None, inputs)[0]
provider = session.get_providers()[0]
if provider == 'CPUExecutionProvider' or result.shape != inputs['image'].shape or not np.isfinite(result).all():
    raise SystemExit('GPU inference verification failed: ' + str(compute.state.snapshot()))
print(provider + ' · ' + adapter[1] + ' · GPU node execution verified')
