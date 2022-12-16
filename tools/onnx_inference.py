import onnx
import onnxruntime

model_path = "condlane_onnx35.onnx"
session = onnxruntime.InferenceSession(model_path, None)

# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# print(input_name)
# print(output_name)