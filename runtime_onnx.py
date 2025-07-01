import onnx, onnxruntime as ort
import numpy as np
from utils import hyper_params_getter, infer_single_image, load_model, infer_single_image_edge, load_onnx_model
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

#model = load_model(hyper_params_getter(), "models/test.ckpt", device="cuda:0")

sess = load_onnx_model("models/test.onnx")


while True:
    dummy_for_pt = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    
    # ONNX Runtime
    start_time = time.time()
    ort_out = infer_single_image_edge(sess, dummy_for_pt, device="cuda:0")
    print("ONNX Runtime inference time:", time.time() - start_time)
    
    # PyTorch
    #start_time = time.time()
    #pt_out = infer_single_image(model, dummy_for_pt, device="cuda:0").cpu().numpy()
    #print("PyTorch inference time:", time.time() - start_time)
    
    #print("Max diff ONNX vs PT:", np.abs(pt_out - ort_out).max())
    # print("Max diff TRT vs PT:", np.abs(pt_out - trt_out).max())
    # print("Max diff TRT vs ONNX:", np.abs(trt_out - ort_out).max())
