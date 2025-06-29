import torch
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from utils import hyper_params_getter


hparams = hyper_params_getter()


if "dinov2" in hparams.backbone_name:
    backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
    train_img_size = (224, 224)
    val_img_size = (322, 322)
    hparams.backbone_name = backbone.backbone_name # in case the user passed dinov2 without the version
    hparams.train_img_size = train_img_size
    hparams.val_img_size = val_img_size
    
elif "resnet" in hparams.backbone_name:
    backbone = ResNet(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, crop_last_block=True)
    train_img_size = (320, 320)
    val_img_size = (384, 384)
    hparams.train_img_size = train_img_size
    hparams.val_img_size = val_img_size
else:
    raise ValueError(f"backbone {hparams.backbone_name} not recognized or not implemented!") 


# Instantiate BoQ aggregator
aggregator = BoQ(
    in_channels=backbone.out_channels,
    proj_channels=hparams.channel_proj,
    num_queries=hparams.num_queries,
    num_layers=hparams.num_layers,
    row_dim=hparams.output_dim//hparams.channel_proj,
    slot_mask=hparams.slot_mask,
    mlp=hparams.mlp,
)

# Define the entire Lightning model for training and validation
model = BoQModel(
    backbone,
    aggregator,
    lr=hparams.lr,
    lr_mul=hparams.lr_mul,
    weight_decay=hparams.weight_decay,
    warmup_epochs=hparams.warmup_epochs,
    milestones=hparams.milestones,
    silent=hparams.silent,
)

model.eval().to('cuda:0')

ckpt = torch.load("models/test.ckpt", map_location="cuda:0")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict)
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    model,                          # 已加载 ckpt 并切到 eval 的模型
    dummy_input,                    # dummy input
    "model.onnx",                   # 输出文件名
    export_params=True,             # 是否导出参数
    opset_version=14,               # ONNX 算子集版本，13+ 通用性好
    do_constant_folding=True,       # 是否对常量折叠优化
    input_names=["input"],          # 输入节点名（可自定义）
    output_names=["output"],        # 输出节点名
    dynamic_axes={                  # 可选：开启动态 batch
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)