import torch
from boq.utils import hyper_params_getter, load_model, build_transform
import numpy as np
import cv2
from PIL import Image

hparams = hyper_params_getter()


model = load_model(hparams, "models/test.ckpt", device="cuda:0")

model.eval().to('cuda:0')

ckpt = torch.load("models/test.ckpt", map_location="cuda:0")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict)

img_input = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
tf = build_transform(model.backbone.backbone_name)
img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
    
dummy_input = tf(img).unsqueeze(0).to("cuda:0")  
torch.onnx.export(
    model,                          
    dummy_input,                    
    "models/test.onnx",                   
    export_params=True,        
    opset_version=17,               
    do_constant_folding=True,    
    input_names=["input"],     
    output_names=["output"],       
    dynamic_axes={                  
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)