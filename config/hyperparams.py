class HyperParams:
    def __init__(self):
        ## Backbone config:
        self.backbone_name: str = "dinov2_vitb14"    # resnet18, resnet50, dinov2_vits14, dinov2_vitl14
        self.unfreeze_n_blocks: int = 2              # number of blocks to unfreeze in the backbone
        
        ## BoQ config:
        self.channel_proj: int = 512
        self.num_queries: int = 64
        self.num_layers: int = 2
        self.output_dim: int = 8192
        self.slot_mask: bool = True  # use slot mask in BoQ
        self.mlp: bool = False  # use MLP for slot mask in BoQ
        
        ## Datasets:
        # NOTE: if you already have OpenVPRLab, you can set the path to the datasets from there
        # otherwise use the dowload scripts in `scripts/` to download to `data/` folder 
        # self.gsv_cities_path: str = "../OpenVPRLab/data/train/gsv-cities"    # path to gsv-cities in OpenVPRLab
        self.gsv_cities_path: str = "./data/train/gsv-cities"                   # or path to gsv-cities in this project
        
        self.cities: str | list = "all" # train on all cities
        # self.cities: str | list = ["Bangkok", "Boston", "PRS"] # train on a subset of cities (check the gsv-cities folder)
        
        self.val_sets: dict = {
            "msls-val":     "./data/val/msls-val",              # path to the msls-val dataset
            "pitts30k-val": "./data/val/pitts30k-val",          # path to the pitts30k-val dataset
        }
        
        ## Training config:
        self.batch_size: int = 64           # batch size is the number of places per batch
        self.img_per_place: int = 4          # number of images per place
        self.max_epochs: int = 40
        self.warmup_epochs: int = 10         # number of linear warmup epochs (not iterations)
        self.lr: float = 1e-4                # learning rate
        self.weight_decay: float = 1e-4
        self.lr_mul: float = 0.1
        self.milestones: list = [10, 20]
        self.num_workers: int = 8
        
        ## misc
        self.silent: bool = False            # disable console output
        self.compile: bool = False           # compile the model using torch.compile() [experimental]
        self.seed: int = 2024                # random seed for reproducibility

        self.dev: bool = False          # enable fast dev run (one train and validation iteration)