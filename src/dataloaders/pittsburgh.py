# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from typing import Optional, Callable, Tuple, Any
import pathlib
import numpy as np
from pathlib import Path
import torchvision
from torch.utils.data import Dataset
from PIL import Image

REQUIRED_FILES = {
    "pitts30k-test":     ["pitts30k_test_dbImages.npy", "pitts30k_test_qImages.npy", "pitts30k_test_gt_25m.npy"],
    "pitts250k-test":    ["pitts250k_test_dbImages.npy", "pitts250k_test_qImages.npy", "pitts250k_test_gt_25m.npy"],
}

class PittsburghDataset(Dataset):

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        transform: Optional[Callable] = None,
        _30k: bool = False,
        _250k: bool = False,
    ):
        
        self.transform = transform
        dataset_path = self._validate_path(dataset_path, _30k, _250k)
        self.dataset_path = dataset_path
        
        # load image names and ground truth data
        self.dbImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][0])
        self.qImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][1])
        self.ground_truth = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][2], allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        # img = Image.open(self.dataset_path / img_path)
        img = torchvision.io.decode_image(self.dataset_path / img_path, mode="RGB")

        if self.transform:
            img = self.transform(img)

        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _validate_path(self, dataset_path, _30k, _250k):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "val" / "pitts"
        path = Path(dataset_path)

        if _30k:
            self.dataset_name = "pitts30k-test"
        elif _250k:
            self.dataset_name = "pitts250k-test"
        else:
            raise ValueError("Either _30k or _250k must be True.")

        msg = "Make sure you downloaded the dataset with the provided script."
        if not path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. {msg}")
        
        if not all((path / file).is_file() for file in REQUIRED_FILES[self.dataset_name]):
            raise FileNotFoundError(f"Missing metadata in {path}. Expected files: {REQUIRED_FILES[self.dataset_name]}")
        
        return path

