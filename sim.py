from onnxsim import simplify
import onnx

onnx_model = onnx.load('./rtmdet_best.onnx')  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, './rtmdet_best_sim.onnx')
print('finished exporting onnx')