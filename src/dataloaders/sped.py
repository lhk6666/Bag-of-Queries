# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision

# NOTE: you can download the this dataset using the scripts we borrowed from 
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# the scripts are in the folder `scripts` in the root of the repository
# this particular dataet comes with hardcoded image names and ground truth
# for faster loading

class SPEDDataset(Dataset):

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        dataset_path = self._validate_path(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.name
        
        # Load image names and ground truth data
        self.dbImages = np.load(dataset_path / "dbImages.npy")
        self.qImages = np.load(dataset_path / "qImages.npy")
        self.ground_truth = np.load(dataset_path / "ground_truth_new.npy", allow_pickle=True)

        # Combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        # img = Image.open(self.dataset_path / img_path).convert("RGB")
        img = torchvision.io.decode_image(self.dataset_path / img_path, mode="RGB")

        if self.transform:
            img = self.transform(img)
            
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _validate_path(self, dataset_path):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "val" / "SPEDTEST"
        path = Path(dataset_path)
        
        msg = "Make sure you downloaded the dataset with the provided script."
        if not path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")
        if not (path / "ref").is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not contain the 'ref' subdirectory. {msg}")
        if not (path / "query").is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not contain the 'query' subdirectory. {msg}")
        if not (path / "dbImages.npy").is_file():
            raise FileNotFoundError(f"The file 'dbImages.npy' does not exist in {path}. {msg}")
        if not (path / "qImages.npy").is_file():
            raise FileNotFoundError(f"The file 'qImages.npy' does not exist in {path}. {msg}")
        if not (path / "ground_truth_new.npy").is_file():
            raise FileNotFoundError(f"The file 'ground_truth_new.npy' does not exist in {path}. {msg}")
        return path