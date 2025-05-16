import torch
import torch
from utils import load_model, infer_single_image, hyper_params_getter, IndexIVFPQ
import time
import os
import shutil

def main(path, ckpt, rank ,start_time):
    hparams = hyper_params_getter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(hparams, ckpt, device)
    emb = infer_single_image(model, path, device)
    ref_embs = torch.load("embeddings/nordland_winter.pt", weights_only=True)
    ref_embs_np = ref_embs.detach().cpu().numpy().astype('float32')
    if ref_embs_np.ndim == 3 and ref_embs_np.shape[1] == 1:
        ref_embs_np = ref_embs_np.reshape(ref_embs_np.shape[0], ref_embs_np.shape[2])
    N, D = ref_embs_np.shape
    print(f"Reshaped reference embeddings to shape: {ref_embs_np.shape}")

    query_np  = emb.detach().cpu().numpy().astype('float32')

    index = IndexIVFPQ(nlist=512, m=16, nbits=8, nprobe=10, k=rank, d=D)

    index.train(ref_embs_np)
    index.add(ref_embs_np)

    distances, indices = index.search(query_np, rank)
    print("Input image number: {}".format(path.split("/")[-1]))
    print("Top-{} image number: {}".format(rank, indices[0] + 1))
    print("Top-{} distances: {}".format(rank, distances[0]))
    end_time = time.time()
    time_taken = end_time - start_time
    image_saver(indices, path, time_taken)

def image_saver(indices,path, time_taken):
    # Create a new directory to store matching images
    
    # Create experiment folder with incremental naming
    experiment_dir = "/home/dragon_llm/daikin/daikin_ws/src/Bag-of-Queries/embeddings/trials"
    dirs = [d for d in os.listdir(experiment_dir) if d.startswith("trial_") and os.path.isdir(os.path.join(experiment_dir, d))]
    next_num = 1
    if dirs:
        nums = [int(d.split("_")[1]) for d in dirs if d.split("_")[1].isdigit()]
        if nums:
            next_num = max(nums) + 1
    
    save_dir = os.path.join(experiment_dir, f"trial_{next_num}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy matched images
    source_dir = "/home/dragon_llm/daikin/daikin_ws/src/VPR-datasets-downloader/datasets/nordland/raw_data/winter"
    for rank, idx in enumerate(indices[0], 1):
        img_name = "images-{:05d}.png".format(idx + 1)  
        rank_img_name = f"{rank}_{img_name}"  # Add rank number to beginning of filename
        source_path = os.path.join(source_dir, img_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(save_dir, rank_img_name))
            print(f"Copied {img_name} to {save_dir} as {rank_img_name}")
        else:
            print(f"Warning: Image {img_name} not found in source directory")
    shutil.copy(path, os.path.join(save_dir, path.split("/")[-1]))
    time_taken_file = os.path.join(save_dir, "time_taken.txt")
    with open(time_taken_file, "w") as f:
        f.write(f"Time taken for the search: {time_taken:.2f} seconds")
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))
    

if __name__ == "__main__":
    ckpt = "logs/dinov2_vitb14/version_0/checkpoints/epoch[19]_R@1[0.9311]_R@5[0.9581].ckpt"
    image_numer = int(input("Please input the image number: "))
    path = "/home/dragon_llm/daikin/daikin_ws/src/VPR-datasets-downloader/datasets/nordland/raw_data/summer/images-{:05d}.png".format(image_numer)
    start_time = time.time()
    main(path, ckpt, rank=5, start_time=start_time)