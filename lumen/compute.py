"""ONNX execution on a real Windows GPU, with a transparent CPU fallback."""
from __future__ import annotations

import ctypes
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from . import performance

log = logging.getLogger(__name__)


class _Luid(ctypes.Structure):
    _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_int32)]


class _Guid(ctypes.Structure):
    _fields_ = [('data1', ctypes.c_uint32), ('data2', ctypes.c_uint16),
                ('data3', ctypes.c_uint16), ('data4', ctypes.c_ubyte * 8)]


class _AdapterDesc(ctypes.Structure):
    _fields_ = [('name', ctypes.c_wchar * 128), ('vendor', ctypes.c_uint32),
                ('device', ctypes.c_uint32), ('subsystem', ctypes.c_uint32),
                ('revision', ctypes.c_uint32), ('dedicated', ctypes.c_size_t),
                ('system', ctypes.c_size_t), ('shared', ctypes.c_size_t),
                ('luid', _Luid), ('flags', ctypes.c_uint32)]


def dxgi_adapters():
    """DirectML device_id is the DXGI adapter index; prefer dedicated VRAM."""
    if os.name != 'nt':
        return []
    factory = ctypes.c_void_p()
    iid = _Guid(0x770aae78, 0xf26f, 0x4dba,
                (ctypes.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
    dll = ctypes.WinDLL('dxgi.dll')
    create = dll.CreateDXGIFactory1
    create.argtypes = (ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        return []

    def method(ptr, index, result, *arguments):
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(vtable[index])

    adapters = []
    try:
        enum = method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(32):
            adapter = ctypes.c_void_p()
            if enum(factory, index, ctypes.byref(adapter)) < 0:
                break
            try:
                desc = _AdapterDesc()
                get_desc = method(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc))
                if get_desc(adapter, ctypes.byref(desc)) >= 0 and not desc.flags & 2:
                    adapters.append((index, desc.name.rstrip('\0'), desc.dedicated, desc.vendor))
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return adapters


def preferred_adapter():
    adapters = dxgi_adapters()
    return max(adapters, key=lambda item: item[2]) if adapters else None


MODES = ('auto', 'cpu', 'dml', 'cuda')


def requested_mode():
    """``LUMEN_COMPUTE`` = auto (default) | cpu | dml | cuda, for troubleshooting and tests."""
    mode = os.environ.get('LUMEN_COMPUTE', 'auto').strip().lower()
    return mode if mode in MODES else 'auto'


def cupy_enabled():
    """CuPy has never been validated on hardware; since 1.3.0 it is opt-in only."""
    return os.environ.get('LUMEN_EXPERIMENTAL_CUPY') == '1' and requested_mode() in ('auto', 'cuda')


def provider_plan(available, adapter, accelerated=True):
    """Ordered (providers, provider, device) attempts before the CPU fallback.

    Execution providers are chosen in one place so a future Windows ML provider
    only has to be added here; DirectML itself is in maintenance mode upstream.
    """
    mode = requested_mode()
    if not accelerated or mode == 'cpu':
        return []
    attempts = []
    if mode in ('auto', 'cuda') and 'CUDAExecutionProvider' in available:
        attempts.append((['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDAExecutionProvider', 'CUDA GPU'))
    if mode in ('auto', 'dml') and adapter is not None and 'DmlExecutionProvider' in available:
        attempts.append(([('DmlExecutionProvider', {'device_id': adapter[0]}), 'CPUExecutionProvider'],
                         'DmlExecutionProvider', adapter[1]))
    return attempts


class ComputeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.provider = 'CPUExecutionProvider'
        self.device = ''
        self.warning = ''
        self.detail = f'{performance.THREADS} 线程 · NumPy / OpenCV'
        self.warning = ''

    def report(self, provider, device='', detail='', warning=''):
        with self.lock:
            self.provider, self.device = provider, device
            self.detail = detail or (f'{performance.THREADS} 线程 · NumPy / OpenCV' if provider == 'CPUExecutionProvider' else provider)
            self.warning = warning

    def snapshot(self):
        with self.lock:
            return self.provider, self.device, self.detail, self.warning


state = ComputeState()


def _gpu_kernels_in_profile(path, provider):
    try:
        events = json.loads(Path(path).read_text(encoding='utf-8'))
        return any(event.get('cat') == 'Node' and event.get('args', {}).get('provider') == provider
                   for event in events)
    finally:
        Path(path).unlink(missing_ok=True)


class Session:
    """Session facade: first inference verifies GPU kernels; failures retry on CPU."""
    def __init__(self, path, accelerated=True):
        import onnxruntime as ort
        self.path = path if isinstance(path, bytes) else str(path)
        self.label = '<in-memory graph>' if isinstance(path, bytes) else Path(self.path).name
        self.ort = ort
        self.provider = 'CPUExecutionProvider'
        self.device = ''
        self.warning = ''
        self._verified = False
        self._lock = threading.Lock()
        available = ort.get_available_providers()
        adapter = preferred_adapter() if accelerated and 'DmlExecutionProvider' in available else None
        attempts = provider_plan(available, adapter, accelerated)
        last_error = ''
        for providers, provider, device in attempts:
            try:
                options = performance.session_options()
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(tempfile.gettempdir()) / 'lumen-dml-profile')
                if provider == 'DmlExecutionProvider':
                    options.enable_mem_pattern = False
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                candidate = ort.InferenceSession(self.path, sess_options=options, providers=providers)
                if provider in candidate.get_providers():
                    self._session, self.provider, self.device = candidate, provider, device
                    break
                Path(candidate.end_profiling()).unlink(missing_ok=True)
            except Exception as exc:
                log.warning('%s session failed for %s', provider, self.label, exc_info=True)
                last_error = str(exc)[:120]
                continue
        else:
            self._session = ort.InferenceSession(self.path,
                sess_options=performance.session_options(), providers=['CPUExecutionProvider'])
            if accelerated:
                self.warning = ('GPU 初始化失败：' + last_error) if last_error else '当前 ONNX 环境没有可用的 GPU 执行设备。'

    def _cpu_fallback(self, warning):
        log.warning('ONNX CPU fallback for %s: %s', getattr(self, 'label', self.path), warning)
        if self.provider != 'CPUExecutionProvider' and not self._verified:
            try:
                Path(self._session.end_profiling()).unlink(missing_ok=True)
            except Exception:
                pass
        self._session = self.ort.InferenceSession(self.path,
            sess_options=performance.session_options(), providers=['CPUExecutionProvider'])
        self.provider, self.device, self._verified = 'CPUExecutionProvider', '', True
        self.warning = warning
        state.report(self.provider, detail=f'{performance.THREADS} 线程 · ONNX CPU', warning=self.warning)

    def run(self, output_names, inputs):
        with self._lock:
            if self.provider == 'CPUExecutionProvider':
                result = self._session.run(output_names, inputs)
                state.report(self.provider, detail=f'{performance.THREADS} 线程 · ONNX CPU',warning=self.warning)
                return result
            if not self._verified:
                try:
                    result = self._session.run(output_names, inputs)
                    path = self._session.end_profiling()
                    if not _gpu_kernels_in_profile(path, self.provider):
                        self._cpu_fallback('此模型没有在 GPU 上执行节点，已切换 CPU。')
                        return self._session.run(output_names, inputs)
                    self._verified = True
                except Exception as exc:
                    self._cpu_fallback(f'GPU 运算失败，已回退 CPU：{str(exc)[:120]}')
                    return self._session.run(output_names, inputs)
            else:
                try:
                    result = self._session.run(output_names, inputs)
                except Exception as exc:
                    self._cpu_fallback(f'GPU 运算失败，已回退 CPU：{str(exc)[:120]}')
                    return self._session.run(output_names, inputs)
            if self.provider not in self._session.get_providers():
                self._cpu_fallback('运行库已将模型切换到 CPU。')
                return result
            state.report(self.provider, self.device, f'{self.provider} · {self.device}')
            return result

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()


def session(path, accelerated=True):
    # Preserve the tiny ONNX stand-in used by the tiling contract test.
    import onnxruntime as ort
    if not hasattr(ort, 'SessionOptions'):
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return Session(path, accelerated)
