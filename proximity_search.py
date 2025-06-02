import torch
import torch
from utils import load_model, infer_single_image, hyper_params_getter, IndexIVFPQ
import time
import os
import glob
import numpy as np
from config.models import ModelName

def main(ckpt, rank, model_name):
    hparams = hyper_params_getter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(hparams, ckpt, device)
    # ref_embs = torch.load("embeddings/nordland_winter_slotmask_2700.pt", weights_only=True)
    ref_embs = torch.load("embeddings/" + model_name + "/nordland_winter.pt", weights_only=True)
    ref_embs_np = ref_embs.detach().cpu().numpy().astype('float32')
    if ref_embs_np.ndim == 3 and ref_embs_np.shape[1] == 1:
        ref_embs_np = ref_embs_np.reshape(ref_embs_np.shape[0], ref_embs_np.shape[2])
    N, D = ref_embs_np.shape
    print(f"Reshaped reference embeddings to shape: {ref_embs_np.shape}")

    index = IndexIVFPQ(nlist=700, m=16, nbits=8, nprobe=100, k=rank, d=D)

    index.train(ref_embs_np)
    index.add(ref_embs_np)
    print(f"Index trained and added {N} reference embeddings with dimension {D}")

    # Process all query images in the folder
    query_folder = "/home/dragon_llm/daikin/daikin_ws/src/boq/image/Nordland/query"

    query_images = sorted(glob.glob(os.path.join(query_folder, "*.jpg")))
    
    print(f"Found {len(query_images)} query images")
    
    # Initialize metrics
    total_time = 0
    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0
    total_queries = len(query_images)
    
    for i, image_path in enumerate(query_images):
        # Extract image number from filename (assuming format images-xxxxx.png)
        image_name = os.path.basename(image_path)
        image_number = int(image_name.split('-')[0].split('.')[0])
        
        # Get query embedding
        emb = infer_single_image(model, image_path, device)
        query_np = emb.detach().cpu().numpy().astype('float32')
        
        # Search
        start_time = time.time()
        distances, indices = index.search(query_np, 10)  # Get top-10 for R@10 calculation
        end_time = time.time()
        
        search_time = end_time - start_time
        total_time += search_time
        
        # Check if ground truth is in top-k results
        # Assuming ground truth is the same image number in reference set
        # Load ground truth mapping if not already loaded
        if 'gt_mapping' not in locals():
            gt_file = "/home/dragon_llm/daikin/daikin_ws/src/boq/image/Nordland/ground_truth_new.npy"
            gt_data = np.load(gt_file, allow_pickle=True)
            gt_mapping = gt_data[:, -1]  # Get the last column with ground truth indices
        
        # Get ground truth indices for current query image
        ground_truth_indices = list(gt_mapping[image_number])
        
        top_indices = indices[0]
        # Check if any ground truth index is in top-k results
        if any(gt_idx in top_indices[:1] for gt_idx in ground_truth_indices):
            correct_r1 += 1
        if any(gt_idx in top_indices[:5] for gt_idx in ground_truth_indices):
            correct_r5 += 1
        if any(gt_idx in top_indices[:10] for gt_idx in ground_truth_indices):
            correct_r10 += 1
        
        if (i + 1) % 100 == 0:
            print(indices)
            print(f"Processed {i + 1}/{total_queries} images")
    
    # Calculate metrics
    avg_time = total_time / total_queries
    r1_score = correct_r1 / total_queries
    r5_score = correct_r5 / total_queries
    r10_score = correct_r10 / total_queries
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries processed: {total_queries}")
    print(f"Average processing time per query: {avg_time:.5f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"R@1: {r1_score:.4f} ({correct_r1}/{total_queries})")
    print(f"R@5: {r5_score:.4f} ({correct_r5}/{total_queries})")
    print(f"R@10: {r10_score:.4f} ({correct_r10}/{total_queries})")
    print("="*50)
    
    # Save results
    result_saver(total_queries, avg_time, total_time, r1_score, r5_score, r10_score, correct_r1, correct_r5, correct_r10)

def result_saver(total_queries, avg_time, total_time, r1_score, r5_score, r10_score, correct_r1, correct_r5, correct_r10):
    experiment_dir = "/home/dragon_llm/daikin/daikin_ws/src/boq/embeddings/trials"
    dirs = [d for d in os.listdir(experiment_dir) if d.startswith("trial_") and os.path.isdir(os.path.join(experiment_dir, d))]
    next_num = 1
    if dirs:
        nums = [int(d.split("_")[1]) for d in dirs if d.split("_")[1].isdigit()]
        if nums:
            next_num = max(nums) + 1
    
    save_dir = os.path.join(experiment_dir, f"trial_{next_num}")
    os.makedirs(save_dir, exist_ok=True)
    
    results_file = os.path.join(save_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("="*50 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Total queries processed: {total_queries}\n")
        f.write(f"Average processing time per query: {avg_time:.5f} seconds\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"R@1: {r1_score:.4f} ({correct_r1}/{total_queries})\n")
        f.write(f"R@5: {r5_score:.4f} ({correct_r5}/{total_queries})\n")
        f.write(f"R@10: {r10_score:.4f} ({correct_r10}/{total_queries})\n")
        f.write("="*50 + "\n")
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    all_models = [name for name in dir(ModelName) if callable(getattr(ModelName, name)) and not name.startswith('__')]
    model_name = input("Please select one model below: " + "\n" + str(all_models) + "\n")
    func = getattr(ModelName, model_name)
    ckpt = func(ModelName)
    main(ckpt, rank=10, model_name=model_name)