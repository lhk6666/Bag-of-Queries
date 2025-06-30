import onnx, onnxruntime as ort
import numpy as np
import torch
import cv2
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from utils import hyper_params_getter, infer_single_image, load_model
import time


model = load_model(hyper_params_getter(), "models/test.ckpt", device="cuda:0")

# 4.1 检查结构合法性
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)


dummy_for_pt = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

dummy_input = torch.from_numpy(dummy_for_pt.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0)
# 4.2 用 onnxruntime 推理比对
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
start_time = time.time()
pt_out = infer_single_image(model, dummy_for_pt, device="cuda:0").cpu().numpy()
print("PyTorch inference time:", time.time() - start_time)
start_time = time.time()
ort_out = sess.run(None, {input_name: dummy_input.cpu().numpy()})[0]
print("ONNX Runtime inference time:", time.time() - start_time)

# 对比最大误差
print("Max diff:", np.abs(pt_out - ort_out).max())
