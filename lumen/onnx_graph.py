"""Dependency-free ONNX writer for the small pointwise GPU graphs.

The runtime graphs only need a few dozen standard operators, so they are encoded
directly as ONNX protobuf messages.  This keeps the `onnx` package out of both
the application and the build environment, and lets the graphs be generated
(and unit tested) from the same Python source on every machine.
"""
from __future__ import annotations

import struct

import numpy as np

FLOAT, INT64 = 1, 7
_DTYPES = {np.dtype(np.float32): FLOAT, np.dtype(np.int64): INT64}


def _varint(value):
    value &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field, wire):
    return _varint(field << 3 | wire)


def _int(field, value):
    return _key(field, 0) + _varint(int(value))


def _bytes(field, data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return _key(field, 2) + _varint(len(data)) + data


def _float(field, value):
    return _key(field, 5) + struct.pack('<f', value)


def tensor(name, array):
    array = np.ascontiguousarray(array)
    body = b''.join(_int(1, d) for d in array.shape)
    body += _int(2, _DTYPES[array.dtype]) + _bytes(8, name) + _bytes(9, array.tobytes())
    return body


def _attribute(name, value):
    body = _bytes(1, name)
    if isinstance(value, float):
        return body + _float(2, value) + _int(20, 1)
    if isinstance(value, int):
        return body + _int(3, value) + _int(20, 2)
    if isinstance(value, (list, tuple)):
        return body + b''.join(_int(8, v) for v in value) + _int(20, 7)
    raise TypeError(value)


def _value_info(name, elem_type, shape):
    dims = b''
    for dim in shape:
        dims += _bytes(1, _bytes(2, dim) if isinstance(dim, str) else _int(1, dim))
    tensor_type = _int(1, elem_type) + _bytes(2, dims)
    return _bytes(1, name) + _bytes(2, _bytes(1, tensor_type))


class Graph:
    """Minimal SSA builder: every operator returns the name of its output."""

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.initializers = {}
        self.inputs = []
        self.outputs = []
        self._count = 0

    def input(self, name, shape, elem_type=FLOAT):
        self.inputs.append(_value_info(name, elem_type, shape))
        return name

    def output(self, value, name, shape):
        self.op('Identity', value, out=name)
        self.outputs.append(_value_info(name, FLOAT, shape))

    def const(self, values, dtype=np.float32):
        array = np.asarray(values, dtype)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.initializers:
            self.initializers[key] = (f'c{len(self.initializers)}', array)
        return self.initializers[key][0]

    def op(self, op_type, *inputs, out=None, **attributes):
        self._count += 1
        out = out or f'{op_type.lower()}_{self._count}'
        names = [self.const(i) if not isinstance(i, str) else i for i in inputs]
        body = b''.join(_bytes(1, n) for n in names) + _bytes(2, out)
        body += _bytes(3, f'n{self._count}') + _bytes(4, op_type)
        body += b''.join(_bytes(5, _attribute(k, v)) for k, v in attributes.items())
        self.nodes.append(body)
        return out

    def serialize(self, opset=13, producer='LUMEN RAW'):
        graph = b''.join(_bytes(1, n) for n in self.nodes) + _bytes(2, self.name)
        graph += b''.join(_bytes(5, tensor(name, array)) for name, array in self.initializers.values())
        graph += b''.join(_bytes(11, i) for i in self.inputs)
        graph += b''.join(_bytes(12, o) for o in self.outputs)
        opset_id = _bytes(1, '') + _int(2, opset)
        return _int(1, 8) + _bytes(2, producer) + _bytes(7, graph) + _bytes(8, opset_id)
