# inference.py
import torch
import os
from tqdm import tqdm
import glob
from utils import load_model, infer_single_image, hyper_params_getter
from config.models import ModelName

def embed_all_images_in_directory(model, directory_path, name, device="cuda:0", output_dir="/home/dragon_llm/daikin/daikin_ws/src/boq/embeddings"):
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
    all_models = [name for name in dir(ModelName) if callable(getattr(ModelName, name)) and not name.startswith('__')]
    model_name = input("Please select one model below: " + "\n" + str(all_models) + "\n")
    func = getattr(ModelName, model_name)
    ckpt = func(ModelName)
    print(f"Loading model {model_name} from checkpoint {ckpt}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(hparams, ckpt, device)
    embed_all_images_in_directory(model, path, name='nordland_winter', device=device, output_dir="/home/dragon_llm/daikin/daikin_ws/src/boq/embeddings/" + model_name)

if __name__ == "__main__":
    nordland_path = "/home/dragon_llm/daikin/daikin_ws/src/boq/image/Nordland/ref"

    main(nordland_path)