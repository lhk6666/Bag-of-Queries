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
        self.ref_embs = None 

    def train(self, ref_embs):
        self.ref_embs = ref_embs.astype('float32') 
        self.index.train(ref_embs)

    def add(self, ref_embs):
        self.index.add(ref_embs)

    def search(self, query_embs, k, rerank=False):
        distances, indices = self.index.search(query_embs, k)
        if rerank and self.ref_embs is not None:
            reranked_indices = []
            reranked_distances = []
            for i in range(query_embs.shape[0]):
                idx = indices[i]
                vecs = self.ref_embs[idx]  # shape: [k, d]
                q = query_embs[i].reshape(1, -1)
                dists = np.linalg.norm(vecs - q, axis=1)
                sort_idx = np.argsort(dists)
                reranked_indices.append(idx[sort_idx])
                reranked_distances.append(dists[sort_idx])
            return np.array(reranked_distances), np.array(reranked_indices)
        return distances, indices

    
def load_onnx_model(ckpt_path: str):
    onnx_model = onnx.load(ckpt_path)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(ckpt_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    return sess

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
        hidden_layer=hparams.hidden_layer,
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
def infer_single_image_edge(sess, img_input):
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

# @torch.no_grad()
# def infer_single_image_trt(engine, img_input):
#     tf = build_transform("dinov2")
#     if isinstance(img_input, str):
#         img = Image.open(img_input).convert("RGB")
#     else:
#         img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#     input_np = tf(img).unsqueeze(0).cpu().numpy().astype(np.float32)

#     # Initialize CUDA context
#     cuda.init()
#     cuda_ctx = cuda.Device(0).make_context()
    
#     try:
#         # Get tensor names
#         io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
#         input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
#         output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
#         input_name = input_names[0]

#         context = engine.create_execution_context()
        
#         # Allocate GPU memory for input
#         d_input = cuda.mem_alloc(input_np.nbytes)
        
#         # Set input shape and allocate memory for all outputs
#         context.set_input_shape(input_name, input_np.shape)
        
#         output_buffers = {}
#         output_shapes = {}
#         for output_name in output_names:
#             output_shape = context.get_tensor_shape(output_name)
#             output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
#             d_output = cuda.mem_alloc(output_size)
#             output_buffers[output_name] = d_output
#             output_shapes[output_name] = output_shape
        
#         # Copy input to GPU
#         cuda.memcpy_htod(d_input, input_np)
        
#         # Set tensor addresses for input and all outputs
#         context.set_tensor_address(input_name, int(d_input))
#         for output_name in output_names:
#             context.set_tensor_address(output_name, int(output_buffers[output_name]))
        
#         # Create CUDA stream
#         stream = cuda.Stream()
        
#         # Execute inference
#         context.execute_async_v3(stream.handle)
#         stream.synchronize()
        
#         # Copy first output back to host (assuming you want the first output)
#         first_output_name = output_names[0]
#         h_output = np.empty(output_shapes[first_output_name], dtype=np.float32)
#         cuda.memcpy_dtoh(h_output, output_buffers[first_output_name])
        
#         # Free GPU memory
#         d_input.free()
#         for d_output in output_buffers.values():
#             d_output.free()
        
#         return h_output
        
#     finally:
#         cuda_ctx.pop()

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
    parser.add_argument("--hidden_layer", type=str, help="Use hidden layer in slot mask MLP.")
    parser.add_argument("--num_layers", type=int, help="Number of layers in the BoQ model.")
    parser.add_argument("--num_queries", type=int, help="Number of queries in the BoQ model.")

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
    if args.num_layers:
        hparams.num_layers = args.num_layers
    if args.num_queries:
        hparams.num_queries = args.num_queries
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
    if args.hidden_layer:
        if args.hidden_layer.lower() == "true":
            hparams.hidden_layer = True
        elif args.hidden_layer.lower() == "false":
            hparams.hidden_layer = False
        else:
            raise ValueError("hidden_layer should be either 'true' or 'false'")
    
    return hparams