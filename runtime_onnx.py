import onnx, onnxruntime as ort
import numpy as np
from utils import hyper_params_getter, infer_single_image, load_model, infer_single_image_edge
import time

model = load_model(hyper_params_getter(), "models/test.ckpt", device="cuda:0")

onnx_model = onnx.load("models/test.onnx")
onnx.checker.check_model(onnx_model)

sess = ort.InferenceSession("models/test.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

while True:
    dummy_for_pt = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    start_time = time.time()
    ort_out = infer_single_image_edge(sess, dummy_for_pt, device="cuda:0")
    print("ONNX Runtime inference time:", time.time() - start_time)
    start_time = time.time()
    pt_out = infer_single_image(model, dummy_for_pt, device="cuda:0").cpu().numpy()
    print("PyTorch inference time:", time.time() - start_time)


    print("Max diff:", np.abs(pt_out - ort_out).max())
