# inference.py
import torch
from PIL import Image
from torchvision import transforms
from lightning.pytorch import Trainer
from src.model import BoQModel
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
import argparse
import os
from tqdm import tqdm
import glob
from config.hyperparams import HyperParams
from utils import load_model, infer_single_image, hyper_params_getter

def embed_all_images_in_directory(model, directory_path, name, device="cuda:0", output_dir="embeddings"):
    emb_list = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files (assuming jpg, png, jpeg extensions)
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    # Process each image
    for img_path in tqdm(sorted(image_files), desc="Embedding images"):
        try:
            
            # Get embedding
            emb = infer_single_image(model, img_path, device)
            emb_list.append(emb)
            # Save embedding
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    all_emb = torch.stack(emb_list, dim=0)
    output_path = os.path.join(output_dir, f"{name}.pt")
    torch.save(all_emb, output_path)

def main(path):
    hparams = hyper_params_getter()
    ckpt = "logs/dinov2_vitb14/version_0/checkpoints/epoch[19]_R@1[0.9311]_R@5[0.9581].ckpt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(hparams, ckpt, device)
    # emb = infer_single_image(model, "image/000000_pitch1_yaw1.jpg", device)
    embed_all_images_in_directory(model, path, name='nordland_winter', device=device)
    # torch.save(emb, "output_embedding.pt")

if __name__ == "__main__":
    nordland_path = "/home/dragon_llm/daikin/daikin_ws/src/VPR-datasets-downloader/datasets/nordland/raw_data/winter"
    main(nordland_path)