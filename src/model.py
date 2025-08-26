# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from pytorch_metric_learning import losses, miners
import torch.nn.functional as F
from src import utils

import matplotlib
matplotlib.use('Agg')  
from matplotlib import pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor

def plot_R_hat(writer, tag, R_hat, global_step):
    # R_hat shape: (Q, Nc) 
    Q, Nc = R_hat.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    R_sample = R_hat.detach().cpu().numpy()  # Shape: (Q, Nc)
    
    im = ax.imshow(R_sample, cmap='viridis', aspect='auto')
    ax.set_title("R_hat Heatmap")
    ax.set_xlabel("Nc Dimension")
    ax.set_ylabel("Query Index")
    plt.colorbar(im, ax=ax)
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)
    
    writer.add_image(tag, image_tensor, global_step)
    
    plt.close(fig)

def plot_similarity_matrix(writer, tag, similarity_matrix, global_step):
    # similarity_matrix shape: (B, patch_size**2)
    B, patch_size_squared = similarity_matrix.shape
    patch_size = int(patch_size_squared ** 0.5)  # Calculate patch_size from patch_size**2
    
    # Select up to 4 samples to plot
    sample_indices = [0, B//4, B//2, 3*B//4] if B >= 4 else list(range(B))
    sample_indices = [idx for idx in sample_indices if idx < B]
    
    cols = 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # Make it easier to index
    
    for i, sample_idx in enumerate(sample_indices):
        # Reshape from (patch_size**2,) to (patch_size, patch_size)
        sim_sample = similarity_matrix[sample_idx].detach().cpu().numpy().reshape(patch_size, patch_size)
        
        ax = axes[i]
        im = ax.imshow(sim_sample, cmap='viridis', aspect='auto')
        ax.set_title(f"Sample {sample_idx}")
        ax.set_xlabel("Patch X")
        ax.set_ylabel("Patch Y")
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].axis('off')
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)
    
    writer.add_image(tag, image_tensor, global_step)
    
    plt.close(fig)

def plot_output_gates(writer, tag, attn_masks, global_step):
    # attn_masks shape: (N, Q, X) where N is batch size, Q is number of queries, X is feature dimension
    if attn_masks.ndim == 3:
        N, Q, X = attn_masks.shape
    elif attn_masks.ndim == 4:
        attn_masks = attn_masks.mean(dim=1)  # Average over heads if needed
        N, Q, X = attn_masks.shape

    # Select up to 4 samples to plot
    sample_indices = [0, N//4, N//2, 3*N//4] if N >= 4 else list(range(N))
    sample_indices = [idx for idx in sample_indices if idx < N]
    
    cols = 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # Make it easier to index
    
    for i, sample_idx in enumerate(sample_indices):
        gate_sample = attn_masks[sample_idx].detach().cpu().numpy()  # Shape: (Q, X)
        
        ax = axes[i]
        im = ax.imshow(gate_sample, cmap='viridis', aspect='auto')
        ax.set_title(f"Sample {sample_idx}")
        ax.set_xlabel("Patch Index")
        ax.set_ylabel("Query Index")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].axis('off')
    
    fig.tight_layout()
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)
    
    writer.add_image(tag, image_tensor, global_step)
    
    plt.close(fig)


def plot_attention_maps(writer, tag, attention_weights, global_step):
    # Select 4 samples from the batch (indices 0, 128, 256, 384)
    batch_size = attention_weights.shape[0]
    sample_indices = [0, batch_size//4, batch_size//2, 3*batch_size//4]
    
    cols = 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # Make it easier to index
    
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx < batch_size:
            attn_sample = attention_weights[sample_idx].detach().cpu()  # Shape: (H, L, S) or (L, S)
            if attn_sample.ndim == 3:
                attn_sample = attn_sample.mean(dim=0)
            ax = axes[i]
            im = ax.imshow(attn_sample, cmap='viridis', aspect='auto')
            ax.set_title(f"Sample {sample_idx}")
            ax.set_xlabel("Key Sequence")
            ax.set_ylabel("Query Sequence")
            fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    # fig.colorbar(im, ax=axes, orientation='horizontal', fraction=.1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)

    writer.add_image(tag, image_tensor, global_step)

    plt.close(fig)

def plot_queries(writer, tag, queries, global_step):
    # queries shape is (Q, C) - no batch dimension
    
    cols = 1
    rows = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    
    query_sample = queries.detach().cpu()  # Shape: (Q, C)
    
    ax = axes if rows * cols == 1 else axes[0]
    im = ax.imshow(query_sample.numpy(), cmap='viridis', aspect='auto')
    ax.set_title(f"Queries Heatmap")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Query Index")
    plt.colorbar(im, ax=ax)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)

    writer.add_image(tag, image_tensor, global_step)

    plt.close(fig)

class BoQModel(L.LightningModule):
    def __init__(
            self, 
            backbone, 
            aggregator,
            lr=1e-4,
            lr_mul=0.1,
            weight_decay=1e-3,
            warmup_epochs=10,
            milestones=[10, 20],
            silent=False,
        ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.lr = lr
        self.lr_mul = lr_mul
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.silent = silent # disable console output
        self.grad_log_interval = 100
        
        # init loss function and miner
        self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)

    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.backbone.parameters(),   "lr": self.lr * 0.1, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warmup learning rate for the first `self.warmup_epochs` epochs
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = (self.trainer.global_step + 1) / total_warmup_steps
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                initial_lr = pg.get("initial_lr", self.lr)
                pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)

    def on_after_backward(self):
        # 只在指定的 step 记录
        if (self.global_step % self.grad_log_interval) != 0:
            return
        writer = self.logger.experiment  # TensorBoard SummaryWriter

        # 1) 全局梯度范数（总览）
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        total_grad_norm = total_norm_sq ** 0.5
        writer.add_scalar("grads/total_grad_norm", total_grad_norm, self.global_step)

        # 2) 关键参数的梯度直方图/均值（示例：Router 的 sigma、Shift）
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            if any(k in name for k in ["router.logits", "cross_attn.in_proj_weight"]):
                writer.add_histogram(f"grads/{name}", p.grad, self.global_step)
                writer.add_scalar(f"grads_mean/{name}", p.grad.mean(), self.global_step)
                writer.add_scalar(f"grads_norm/{name}", p.grad.data.norm(2), self.global_step)

        # 3) 可选：检测 NaN/Inf
        has_nan = any([torch.isnan(p.grad).any().item() for p in self.parameters() if p.grad is not None])
        writer.add_scalar("grads/has_nan", float(has_nan), self.global_step)


    
    @torch.compiler.disable()
    def compute_loss(self, descriptors, labels):
        mined_pairs = self.ms_miner(descriptors, labels)
        loss =  self.ms_loss(descriptors, labels, mined_pairs)
        return loss
    
    def forward(self, x):
        x, cls = self.backbone(x)
        x, attns, attn_masks, queries, R, s = self.aggregator(x, cls)
        return x, attns, queries, attn_masks, R, s

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images.shape is (P, K, C, H, W) with P: number of places, K: number of views per place
        # labels.shape is (P, K)
        images = images.flatten(0, 1) # P*K, C, H, W 
        labels = labels.flatten() # P*K
        
        # forward pass
        descriptors, attentions, queries, attn_masks, R, s = self(images)
        B, _ = descriptors.shape  
        # queries = torch.cat(queries, dim=1)
        # compute loss
        loss = self.compute_loss(descriptors, labels)
        # kl_loss = self.compute_attn_mask_loss(attentions, attn_masks, B)
        # diversity = self.query_diversity_loss(queries[0])
        if self.trainer.global_step % self.grad_log_interval == 0 and not self.silent:
            # log attention maps for the first batch
            if isinstance(attentions, (list, tuple)):
                for i, attn_tensor in enumerate(attentions):
                    # attn_tensor should be the actual (N, H, L, S) tensor
                    plot_attention_maps(
                        writer=self.logger.experiment,
                        tag=f"attention_maps/layer_{i+1}",  # Use a unique tag for each layer!
                        attention_weights=attn_tensor,
                        global_step=self.trainer.global_step
                    )
            else: # If attentions is just a single tensor
                plot_attention_maps(
                    writer=self.logger.experiment,
                    tag="attention_maps/single_layer",
                    attention_weights=attentions,
                    global_step=self.trainer.global_step
                )
            for i, query_tensor in enumerate(queries):
                plot_queries(
                    writer=self.logger.experiment,
                    tag=f"queries/layer_{i+1}",
                    queries=query_tensor[0],
                    global_step=self.trainer.global_step
                )
            if attn_masks is not None and len(attn_masks) > 0:
                # Move each mask to CPU first, then concatenate to avoid GPU memory issues
                attn_masks_cpu = [mask.squeeze(-1).detach().cpu() for mask in attn_masks]
                attn_masks = torch.cat(attn_masks_cpu, dim=0)
                plot_output_gates(
                    writer=self.logger.experiment,
                    tag="output_gates",
                    attn_masks=attn_masks,
                    global_step=self.trainer.global_step
                )
            if R is not None and s is not None:
                # log the R_hat and slot masks
                for i, r in enumerate(R):
                    plot_R_hat(
                        writer=self.logger.experiment,
                        tag=f"R_hat/layer_{i+1}",
                        R_hat=r,
                        global_step=self.trainer.global_step
                    )
                for i, similar in enumerate(s):
                    if similar is not None:
                        plot_similarity_matrix(
                            writer=self.logger.experiment,
                            tag=f"similarity_matrix/layer_{i+1}",
                            similarity_matrix=similar,
                            global_step=self.trainer.global_step
                        )
                    

        self.log("loss", loss, prog_bar=True, logger=True)
        # self.log("kl_loss", kl_loss, prog_bar=True, logger=True)
        return loss 
    
    def on_train_epoch_end(self):
        # reload the dataframes to shuffle in-city
        # this is faster than reloading the entire dataloader
        self.trainer.train_dataloader.dataset._refresh_dataframes()
        # for i, mask in enumerate(self.aggregator.masks):
        #     mask = mask.detach().cpu().numpy()
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(f"Slot Mask {i}")
        #     plt.colorbar()
        #     plt.savefig(f"slot_mask_epoch{self.current_epoch}_mask{i}.png")
        #     plt.close()
        
    def on_validation_epoch_start(self):
        # we init an empty dictionary to store the descriptors for each dataloader
        self.validation_outputs = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        descriptors, _, _, _, _, _ = self(images)
        descriptors = descriptors.detach().cpu()#.numpy()
        
        if dataloader_idx not in self.validation_outputs:
            # keep in mind that we might have multiple validation dataloaders
            # initialize an empty list for this dataloader, then append the descriptors
            self.validation_outputs[dataloader_idx] = []
            
        # save the descriptors to compute the recall@k at the end of the validation epoch
        self.validation_outputs[dataloader_idx].append(descriptors)

    def on_validation_epoch_end(self):
        # get the validation dataloaders        
        val_dataloaders = self.trainer.val_dataloaders
        recalls = {} # one dict for each validation set
        for dataloader_idx, descriptors_list in self.validation_outputs.items():
            descriptors = torch.cat(descriptors_list, dim=0)
            dataset = val_dataloaders[dataloader_idx].dataset

            if self.trainer.fast_dev_run:
                # skip the recall computation for fast dev runs
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
            else:
                # we will use the descriptors, the number of references, number of queries, and the ground truth
                # NOTE: make sure these are available in the dataset object and ARE IN THE RIGHT ORDER.
                # meaning that the first `num_references` descriptors are reference images and the rest are query images
                recalls_dict = utils.compute_recall_performance(
                        descriptors, 
                        dataset.num_references,
                        dataset.num_queries,
                        dataset.ground_truth,
                        k_values=[1, 5, 10, 15],
                )
                recalls_log = {
                    f"{dataset.dataset_name}/R@1": recalls_dict[1],
                    f"{dataset.dataset_name}/R@5": recalls_dict[5],
                }
                recalls[dataset.dataset_name] = recalls_dict
                
                # add to the logger but not the progress bar 
                # we will display the results below
                self.log_dict(recalls_log, prog_bar=False, logger=True)
        
        if recalls and not self.silent:
            utils.display_recall_performance(
                list(recalls.values()), 
                list(recalls.keys()),
            )
        self.validation_outputs.clear()