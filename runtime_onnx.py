import onnx, onnxruntime as ort
import numpy as np
import torch
import cv2
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from utils import hyper_params_getter, infer_single_image, load_model, build_transform, infer_single_image_edge
import time
from PIL import Image

model = load_model(hyper_params_getter(), "models/test.ckpt", device="cuda:0")

onnx_model = onnx.load("models/test.onnx")
onnx.checker.check_model(onnx_model)


dummy_for_pt = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

tf = build_transform(model.backbone.backbone_name)
img = cv2.cvtColor(dummy_for_pt, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
    
# dummy_input = tf(img).unsqueeze(0).cpu().numpy().astype(np.float32)


sess = ort.InferenceSession("models/test.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

input_name = sess.get_inputs()[0].name

while True:
    start_time = time.time()
    ort_out = infer_single_image_edge(dummy_for_pt, device="cuda:0")
    print("ONNX Runtime inference time:", time.time() - start_time)
    start_time = time.time()
    pt_out = infer_single_image(model, dummy_for_pt, device="cuda:0").cpu().numpy()
    print("PyTorch inference time:", time.time() - start_time)


    print("Max diff:", np.abs(pt_out - ort_out).max())
