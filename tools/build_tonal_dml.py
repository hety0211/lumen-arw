"""Regenerate lumen/tonal_dml.py; requires onnx only on the developer machine."""
import base64
from pathlib import Path
import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

nodes = []
def op(name, inputs, output):
    nodes.append(helper.make_node(name, inputs, [output]))
    return output

constants = {}
def c(name, values):
    constants[name] = numpy_helper.from_array(np.asarray(values, np.float32), name)
    return name

c('luma', np.array([.2126, .7152, .0722], np.float32).reshape(1, 3, 1, 1))
for name, value in [('zero', 0), ('one', 1), ('two', 2), ('three', 3),
                    ('six', 6), ('power', .45), ('sixtyfive', 65),
                    ('eightyfive', 85), ('threshold', .0031308),
                    ('gamma', 1 / 2.4), ('slope', 12.92), ('srgbscale', 1.055),
                    ('offset', .055), ('half', .5)]:
    c(name, [value])

op('Mul', ['image', 'gains'], 'balanced')
op('Mul', ['balanced', 'exposure'], 'exposed')
op('Mul', ['exposed', 'luma'], 'weighted')
nodes.append(helper.make_node('ReduceSum', ['weighted'], ['luminance'], axes=[1], keepdims=1))
op('Clip', ['luminance', 'zero', 'one'], 'clamped_luma')
op('Pow', ['clamped_luma', 'power'], 'p')
op('Sub', ['one', 'p'], 'inv_p')
op('Pow', ['inv_p', 'two'], 'sw')
op('Pow', ['p', 'three'], 'hw')
op('Mul', ['shadows', 'sw'], 'shadow_gain')
op('Mul', ['highlights', 'hw'], 'highlight_gain')
op('Add', ['shadow_gain', 'highlight_gain'], 'middle_sum')
op('Div', ['middle_sum', 'sixtyfive'], 'middle_stops')
op('Pow', ['inv_p', 'six'], 'bw')
op('Pow', ['p', 'six'], 'ww')
op('Mul', ['blacks', 'bw'], 'black_gain')
op('Mul', ['whites', 'ww'], 'white_gain')
op('Add', ['black_gain', 'white_gain'], 'end_sum')
op('Div', ['end_sum', 'eightyfive'], 'end_stops')
op('Add', ['middle_stops', 'end_stops'], 'stops')
op('Pow', ['two', 'stops'], 'tonal_gain')
op('Mul', ['exposed', 'tonal_gain'], 'linear')
op('Max', ['linear', 'zero'], 'positive')
op('LessOrEqual', ['positive', 'threshold'], 'low_mask')
op('Mul', ['positive', 'slope'], 'low_srgb')
op('Pow', ['positive', 'gamma'], 'high_gamma')
op('Mul', ['high_gamma', 'srgbscale'], 'high_scaled')
op('Sub', ['high_scaled', 'offset'], 'high_srgb')
op('Where', ['low_mask', 'low_srgb', 'high_srgb'], 'srgb')
op('Sub', ['srgb', 'half'], 'centered')
op('Mul', ['centered', 'contrast'], 'contrasted')
op('Add', ['contrasted', 'half'], 'lifted')
op('Clip', ['lifted', 'zero', 'one'], 'output')

value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
inputs = [value('image', [1, 3, 'height', 'width']), value('gains', [1, 3, 1, 1])]
inputs += [value(name, [1]) for name in
           ('exposure', 'shadows', 'highlights', 'blacks', 'whites', 'contrast')]
graph = helper.make_graph(nodes, 'Lumen tonal pipeline', inputs,
                          [value('output', [1, 3, 'height', 'width'])],
                          initializer=list(constants.values()))
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)],
                          producer_name='LUMEN RAW')
model.ir_version = 10
onnx.checker.check_model(model)
encoded = base64.b64encode(model.SerializeToString()).decode('ascii')
target = Path(__file__).resolve().parents[1] / 'lumen' / 'tonal_dml.py'
target.write_text('"""Generated ONNX graph for GPU tonal rendering; see tools/build_tonal_dml.py."""\n'
                  'import base64\nMODEL = base64.b64decode(\n'
                  + '\n'.join(f'    {encoded[i:i+92]!r}' for i in range(0, len(encoded), 92))
                  + '\n)\n', encoding='utf-8')
print(f'{target}: {len(model.SerializeToString())} bytes')
