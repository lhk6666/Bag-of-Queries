# inference.py
import torch
import os
from tqdm import tqdm
import glob
from boq.utils import load_model, infer_single_image, hyper_params_getter
from boq.config.models import ModelName
from pathlib import Path
import re

def embed_all_images_in_directory(model, directory_path, name, device="cuda:0", output_dir="/home/dragon_llm/daikin/daikin_ws/src/boq/embeddings", posi_embed_flag=False):
    emb_list = []
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    for img_path in tqdm(sorted(image_files), desc="Embedding images"):
        try:
            emb = infer_single_image(model, img_path, device)
            if posi_embed_flag:
                img_name = os.path.basename(img_path)
                pattern = r"x([-\d\.]+)_y([-\d\.]+)_z([-\d\.]+)_roll([-\d\.]+)_pitch([-\d\.]+)_yaw([-\d\.]+)"
                match = re.search(pattern, img_name)
                if match:
                    x, y, z, roll, pitch, yaw = [float(val.rstrip('.')) for val in match.groups()]
                    tensor_pose = torch.tensor([x, y, z, roll, pitch, yaw], dtype=torch.float32)
                    tensor_pose = tensor_pose.unsqueeze(0)
                    emb = torch.cat((emb, tensor_pose), dim=1)
            emb_list.append(emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    all_emb = torch.stack(emb_list, dim=0)
    output_path = os.path.join(output_dir, f"{name}.pt")
    print(f"Saving embeddings with shape {all_emb.shape} to {output_path}")
    torch.save(all_emb, output_path)

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    posi_embed_flag = False
    dataset_name = input("Choose dataset name: \n1. nordland_winter\n2. 0066_3_3_1\n3. daikin_factory\n4. utokyo_eng8_-2-1\n")
    if dataset_name == "nordland_winter" or dataset_name == "1":
        path = os.path.join(current_dir, "image/Nordland/ref")
        name = "nordland_winter"
    elif dataset_name == "0066_3_3_1" or dataset_name == "2": 
        path = os.path.join(current_dir, "image/0066_3_3_1/0066_3*3*1")
        name = "0066_3_3_1"
    elif dataset_name == "daikin_factory" or dataset_name == "3":
        path = "/media/dragon_llm/3C09549315E08290/pointcloud/images"
        name = "daikin_factory"
    elif dataset_name == "utokyo_eng8_-2-1" or dataset_name == "4":
        path = os.path.join(current_dir, "image/utokyo_eng8_-2-1")
        name = "utokyo_eng8_-2-1"
        posi_embed_flag = True
    else:
        raise ValueError("Invalid dataset name. Please choose either 'nordland_winter' or '0066_3_3_1' or 'daikin_factory' or 'utokyo_eng8_-2-1'.")

    hparams = hyper_params_getter()
    all_models = [name for name in dir(ModelName) if callable(getattr(ModelName, name)) and not name.startswith('__')]
    model_name = input("Please select one model below: " + "\n" + str(all_models) + "\n")
    func = getattr(ModelName, model_name)
    ckpt = func(ModelName)
    print(f"Loading model {model_name} from checkpoint {ckpt}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(hparams, ckpt, device)
    # embed_all_images_in_directory(model, path, name='nordland_winter', device=device, output_dir="embeddings/" + model_name)
    embed_all_images_in_directory(model, path, name=name, device=device, output_dir=os.path.join(current_dir, "embeddings", model_name), posi_embed_flag=posi_embed_flag)