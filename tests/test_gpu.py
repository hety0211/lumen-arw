"""GPU graph parity and status handling; the actual DirectML run is a hardware check."""
import numpy as np
import pytest
import sys
import threading
import types

from lumen import compute, engine, model


def test_tonal_graph_matches_cpu_across_adjustments():
    rng = np.random.default_rng(122)
    image = rng.random((57, 83, 3), dtype=np.float32) * 1.3
    image[0, :3] = 0
    image[1, :3] = 1
    edits = model.adjustments()
    edits.update(temperature=37, tint=-24, exposure=.7, shadows=47,
                 highlights=-38, blacks=-21, whites=28, contrast=36)
    cpu = engine.Backend('cpu').tonal(image, edits)
    gpu_graph = engine.Backend('cpu')._tonal_dml(image, edits)
    np.testing.assert_allclose(gpu_graph, cpu, rtol=2e-5, atol=3e-6)


@pytest.mark.skipif(sys.platform == 'darwin', reason='DXGI adapters exist only on Windows; macOS has one Metal device')
def test_preferred_adapter_uses_dedicated_memory(monkeypatch):
    monkeypatch.setattr(compute, 'dxgi_adapters', lambda: [
        (0, 'integrated', 512_000_000, 0x1002),
        (1, 'dedicated', 16_000_000_000, 0x1002)])
    assert compute.preferred_adapter()[0] == 1
    monkeypatch.setenv('LUMEN_DML_DEVICE', '0')
    assert compute.preferred_adapter()[1] == 'integrated'
    monkeypatch.setenv('LUMEN_DML_DEVICE', '7')
    assert compute.preferred_adapter()[0] == 1


def test_device_failure_retries_on_cpu():
    class FailedGpu:
        def run(self, *_):
            raise RuntimeError('device lost')

    class Cpu:
        def run(self, *_):
            return [np.array([42], np.float32)]

    session = compute.Session.__new__(compute.Session)
    session.path = 'unused.onnx'
    session.ort = types.SimpleNamespace(InferenceSession=lambda *a, **k: Cpu())
    session._session = FailedGpu()
    session._lock = threading.Lock()
    session._verified = True
    session.provider = 'DmlExecutionProvider'
    session.device = 'GPU'
    session.warning = ''
    result = session.run(None, {})[0]
    assert result[0] == 42
    assert session.get_providers() == ['CPUExecutionProvider']
    assert 'device lost' in compute.state.snapshot()[3]
