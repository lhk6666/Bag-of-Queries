import torch
import torch
from utils import load_model, infer_single_image, hyper_params_getter, IndexIVFPQ, infer_single_image_edge, load_onnx_model
import time
import os
import glob
import numpy as np
from config.models import ModelName

class ProximitySearcher:
    def __init__(self, ckpt, model_name, device=None, use_onnx=False):
        self.hparams = hyper_params_getter()
        self.use_onnx = use_onnx
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        if not use_onnx:
            self.model = load_model(self.hparams, ckpt, self.device)
        else:
            self.model = load_onnx_model(ckpt)
        self.model_name = model_name
        self.index = None
        self.ref_embs_np = None
        self.gt_mapping = None
        
    def load_reference_embeddings(self, embedding_path=None):
        if embedding_path is None:
            embedding_path = f"embeddings/{self.model_name}/nordland_winter.pt"
        
        ref_embs = torch.load(embedding_path, weights_only=True)
        self.ref_embs_np = ref_embs.detach().cpu().numpy().astype('float32')
        
        if self.ref_embs_np.ndim == 3 and self.ref_embs_np.shape[1] == 1:
            self.ref_embs_np = self.ref_embs_np.reshape(self.ref_embs_np.shape[0], self.ref_embs_np.shape[2])
        
        N, D = self.ref_embs_np.shape
        print(f"Loaded reference embeddings with shape: {self.ref_embs_np.shape}")
        return N, D
    
    def build_index(self, nlist=700, m=16, nbits=8, nprobe=100, k=10):
        if self.ref_embs_np is None:
            raise ValueError("Please load reference embeddings first")
        
        N, D = self.ref_embs_np.shape
        self.index = IndexIVFPQ(nlist=nlist, m=m, nbits=nbits, nprobe=nprobe, k=k, d=D)
        self.index.train(self.ref_embs_np)
        self.index.add(self.ref_embs_np)
        print(f"Index trained and added {N} reference embeddings with dimension {D}")
    
    def load_ground_truth(self, gt_file=None):
        if gt_file is None:
            gt_file = "image/Nordland/ground_truth_new.npy"
        
        gt_data = np.load(gt_file, allow_pickle=True)
        self.gt_mapping = gt_data[:, -1]
        print(f"Loaded ground truth mapping with {len(self.gt_mapping)} entries")
    
    def search_single_image(self, image_path=None, image=None, top_k=10, rerank=False):
        if self.index is None:
            raise ValueError("Please build index first")
        
        start_time = time.time()
        
        if image_path is not None:
            if not self.use_onnx:
                emb = infer_single_image(self.model, image_path, self.device)
            else:
                emb = infer_single_image_edge(self.model, image_path)
        elif image is not None:
            if not self.use_onnx:
                emb = infer_single_image(self.model, image, self.device)
            else:
                emb = infer_single_image_edge(self.model, image)
        query_np = emb.detach().cpu().numpy().astype('float32') if not self.use_onnx else emb.astype('float32')
        
        distances, indices = self.index.search(query_np, top_k, rerank=rerank)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        return {
            'distances': distances[0],
            'indices': indices[0],
            'search_time': search_time
        }
    
    def evaluate_single_image(self, image_path, image_number=None):
        if self.gt_mapping is None:
            self.load_ground_truth()
        
        if image_number is None:
            image_name = os.path.basename(image_path)
            image_number = int(image_name.split('-')[0].split('.')[0])
        
        result = self.search_single_image(image_path, top_k=10)
        
        ground_truth_indices = list(self.gt_mapping[image_number])
        
        top_indices = result['indices']
        r1 = any(gt_idx in top_indices[:1] for gt_idx in ground_truth_indices)
        r5 = any(gt_idx in top_indices[:5] for gt_idx in ground_truth_indices)
        r10 = any(gt_idx in top_indices[:10] for gt_idx in ground_truth_indices)
        
        return {
            'search_result': result,
            'ground_truth_indices': ground_truth_indices,
            'r1': r1,
            'r5': r5,
            'r10': r10,
            'image_number': image_number
        }

def batch_evaluation(searcher, query_folder, rank=10):
    query_images = sorted(glob.glob(os.path.join(query_folder, "*.jpg")))
    print(f"Found {len(query_images)} query images")
    
    total_time = 0
    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0
    total_queries = len(query_images)
    
    for i, image_path in enumerate(query_images):
        result = searcher.evaluate_single_image(image_path)
        
        total_time += result['search_result']['search_time']
        if result['r1']:
            correct_r1 += 1
        if result['r5']:
            correct_r5 += 1
        if result['r10']:
            correct_r10 += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_queries} images")
    
    avg_time = total_time / total_queries
    r1_score = correct_r1 / total_queries
    r5_score = correct_r5 / total_queries
    r10_score = correct_r10 / total_queries
    
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
    
    result_saver(total_queries, avg_time, total_time, r1_score, r5_score, r10_score, correct_r1, correct_r5, correct_r10)

def result_saver(total_queries, avg_time, total_time, r1_score, r5_score, r10_score, correct_r1, correct_r5, correct_r10):
    experiment_dir = "embeddings/trials"
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

def main(ckpt, rank, model_name, use_onnx=False):
    searcher = ProximitySearcher(ckpt, model_name, use_onnx=use_onnx)
    searcher.load_reference_embeddings()
    searcher.build_index(k=rank)
    
    query_folder = "image/Nordland/query"
    batch_evaluation(searcher, query_folder, rank)

def single_image_demo(ckpt, model_name, image_path=None, image=None, use_onnx=False):
    searcher = ProximitySearcher(ckpt, model_name, use_onnx=use_onnx)
    searcher.load_reference_embeddings(f"embeddings/{model_name}/0066_3_3_1.pt")
    searcher.build_index(nlist=20, m=4, nbits=8, nprobe=10, k=10)
    
    if image is not None:
        result = searcher.search_single_image(image=image, top_k=10)
    elif image_path is not None:
        result = searcher.search_single_image(image_path=image_path, top_k=10)

    return result['indices'], result['distances']

if __name__ == "__main__":
    all_models = [name for name in dir(ModelName) if callable(getattr(ModelName, name)) and not name.startswith('__')]
    model_name = input("Please select one model below: " + "\n" + str(all_models) + "\n")
    func = getattr(ModelName, model_name)
    ckpt = func(ModelName)
    use_onnx = input("Use ONNX model? (yes/no): ").strip().lower() == "yes"
    if use_onnx:
        ckpt = ckpt.replace(".ckpt", ".onnx")
    
    mode = input("Select mode: 1 for batch evaluation, 2 for single image demo: ")
    
    if mode == "1":
        main(ckpt, rank=10, model_name=model_name, use_onnx=use_onnx)
    elif mode == "2":
        image_path = input("Enter image path: ")
        indices, distance = single_image_demo(ckpt, model_name, image_path, use_onnx=use_onnx)
        print(f"Indices: {indices}")
        print(f"Distances: {distance}")
    else:
        print("Invalid mode selected")