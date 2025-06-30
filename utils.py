from src.model import BoQModel
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
import torch
from torchvision import transforms
from PIL import Image
import argparse
from config.hyperparams import HyperParams
import faiss
import cv2
import onnx, onnxruntime as ort
import numpy as np

class IndexIVFPQ():
    def __init__(self, nlist, m, nbits, nprobe, k , d):
        self.quantizer = faiss.IndexFlatL2(d)
        self.d = d
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = faiss.IndexIVFPQ(self.quantizer, d, nlist, m, nbits)
        self.index.use_precomputed_table = True
        self.index.nprobe = nprobe
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def train(self, ref_embs):
        self.index.train(ref_embs)

    def add(self, ref_embs):
        self.index.add(ref_embs)

    def search(self, query_embs, k):
        distances, indices = self.index.search(query_embs, k)
        return distances, indices

def load_model(hparams, ckpt_path: str, device: str = "cuda:0"):
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
    model = BoQModel.load_from_checkpoint(
        ckpt_path,
        backbone=backbone,
        aggregator=aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )
    model = model.to(device)
    model.eval()
    return model

def build_transform(backbone_name: str):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

@torch.no_grad()
def infer_single_image(model: BoQModel, img_input, device: str = "cuda:0"):
    tf = build_transform(model.backbone.backbone_name)
    
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    
    x = tf(img).unsqueeze(0).to(device)   # [1,3,H,W]
    output = model(x)
    embedding = output[0] if isinstance(output, tuple) else output  
    return embedding.cpu()

@torch.no_grad()
def infer_single_image_edge(sess, img_input, device: str = "cuda:0"):
    tf = build_transform("dinov2")
    
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    
    dummy_input = tf(img).unsqueeze(0).cpu().numpy().astype(np.float32)


    # sess = ort.InferenceSession("models/test.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: dummy_input})[0] 
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    parser.add_argument("--dev",      action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--silent",   action="store_true", help="Disable console output.")
    parser.add_argument('--compile',  action='store_true', help='Compile the model using torch.compile()')
    
    parser.add_argument("--seed",   type=int,   help="Random seed for reproducibility.")
    
    parser.add_argument("--bs",     type=int,   help="Batch size.")
    parser.add_argument("--lr",     type=float, help="Learning Rate.")
    parser.add_argument("--wd",     type=float, help="Weight Decay.")
    
    parser.add_argument('--epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--warmup', type=int, help='Number of warmup epochs')
    parser.add_argument("--nw",     type=int, help="Numbers of workers.")

    parser.add_argument('--backbone',   type=str, help='Backbone model name [resnet50, dinov2]')
    parser.add_argument('--unfreeze_n', type=int, help='Number of blocks to unfreeze in the backbone.')
    parser.add_argument("--dim",        type=int, help="Output dimensionality.")

    parser.add_argument("--slotmask", type=str, help="Slot mask for the model.")
    parser.add_argument("--mlp", type=str, help="Use MLP for slot mask in BoQ.")

    return parser.parse_args()

def hyper_params_getter():
    args = parse_args()
    hparams = HyperParams()
    
    if args.seed:
        hparams.seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.bs:
        hparams.batch_size = args.bs
    if args.lr:
        hparams.lr = args.lr
    if args.wd:
        hparams.weight_decay = args.wd
    if args.epochs:
        hparams.max_epochs = args.epochs
    if args.warmup:
        hparams.warmup_epochs = args.warmup
    if args.nw:
        hparams.num_workers = args.nw
    if args.backbone:
        hparams.backbone_name = args.backbone
    if args.unfreeze_n:
        hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim:
        hparams.output_dim = args.dim
    if args.dev:
        hparams.dev_mode = args.dev
    if args.slotmask:
        if args.slotmask.lower() == "true":
            hparams.slot_mask = True
        elif args.slotmask.lower() == "false":
            hparams.slot_mask = False
        else:
            raise ValueError("slotmask should be either 'true' or 'false'")
    if args.mlp:
        if args.mlp.lower() == "true":
            hparams.mlp = True
        elif args.mlp.lower() == "false":
            hparams.mlp = False
        else:
            raise ValueError("mlp should be either 'true' or 'false'")
    
    return hparams