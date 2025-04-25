import torch
import torch
from utils import load_model, infer_single_image, hyper_params_getter, IndexIVFPQ

def main(path, ckpt, rank):
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
    print("Top-{} indices: {}".format(rank, indices[0]))
    print("Top-{} distances: {}".format(rank, distances[0]))

if __name__ == "__main__":
    ckpt = "logs/dinov2_vitb14/version_0/checkpoints/epoch[19]_R@1[0.9311]_R@5[0.9581].ckpt"
    path = "/home/dragon_llm/daikin/daikin_ws/src/VPR-datasets-downloader/datasets/nordland/raw_data/summer/images-11210.png"
    main(path, ckpt, rank=5)
