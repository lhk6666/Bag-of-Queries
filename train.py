# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/boq
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.datamodule import VPRDataModule
from utils import hyper_params_getter

def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)
    torch.set_float32_matmul_precision("high") 
    
    # Instantiate the backbone and define the image size for training and validation
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, output_layers=hparams.out_layers)
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
        in_channels=[backbone.out_channels for _ in range(len(backbone.output_layers))],
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim//hparams.channel_proj,
        slot_mask=hparams.slot_mask,
        mlp=hparams.mlp,
        hidden_layer=hparams.hidden_layer,
    )
    
    # Define the entire Lightning model for training and validation
    model = BoQModel(
        backbone,
        aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )
    
    if hparams.compile:
        model = torch.compile(model)
    
    
    
    # Define the datamodule for handling training and validation datasets
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )
    
    # If you want to display the datasets and training configs
    if not hparams.silent:
        datamodule.setup()                  # first init the datasets
        display_datasets_stats(datamodule)  # then display the stats
    
    # we use Tensorboard for logging (integrated with PyTorch Lightning)
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False
    )
    
    # let's save all the hyperparameters to the the log file
    # this will be saved in the logs folder
    # e.g. ./logs/dinov2_vitb14/version_0/hparams.yaml
    tensorboard_logger.log_hyperparams(hparams.__dict__) 
    
    # Define the checkpointing callback
    checkpointing = callbacks.ModelCheckpoint(
        monitor="msls-val/R@1",  # <==== monitor the Recall@1 on the msls-val dataset
        filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()
    
    # Lightning Trainer will take a list of callbacks
    callback_list = [checkpointing]
    if not hparams.silent:
        callback_list.append(program_bar)
    
    # Define the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,          
        # precision="16-mixed",
        precision=hparams.precision,  # "16-mixed" or "32"
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
    )
    
    # Train the model
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    hparams = hyper_params_getter()
    
    train(hparams, dev_mode=hparams.dev)