import codecs
import sys, os, warnings
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
import numbers
from collections.abc import Sequence
import torchvision.transforms as transforms
import kornia as K

def _setup_size(size, center, error_msg):
    if not isinstance(center, tuple):
        raise ValueError(error_msg)

    if isinstance(size, numbers.Number):
        return int(center[1]-(int(size)/2)), int(center[0]-(int(size)/2)), int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return int(center[1]-(size[0]/2)), int(center[0]-(size[0]/2)), size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return int(center[1]-(size[0]/2)), int(center[0]-(size[1]/2)), size[0], size[1]

class DongjinCrop(torch.nn.Module):
    """Crops the given image at given point
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        center (tuple of int): the center point to crop. If provided coordinates exceeds image sizes, image with all 0 will be returned.
    """
    def __init__(self, size, center):
        super().__init__()
        self.size = _setup_size(size, center, error_msg="Please make sure that center is tuple of (x, y) and to provide only two dimensions (h, w) for size.")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return transforms.functional.crop(img, *self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class DongjinTransform():
    def __init__(self, img_resize):
        self.img_resize = img_resize

    def get(self,):
        tf = transforms.Compose([
            DongjinCrop(self.img_resize, (1300, 600)),
            transforms.Grayscale(),
            # transforms.ColorJitter(brightness=.5, hue=.3, saturation=.3, contrast=.5),
            # transforms.RandomAffine(degrees=360, scale=(0.1, 1.1)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=(0,180)),
            transforms.ToTensor(),
            # Sobel Edge Detection
            # transforms.Lambda(lambda x: (1. - K.filters.sobel(x.unsqueeze(0))).squeeze(0)),
            # Laplacian Edge Detection
            transforms.Lambda(lambda x: (1. - K.filters.laplacian(x.unsqueeze(0), kernel_size=5).clamp(0., 1.)).squeeze(0)),
            # Canny Edge Detection
            # transforms.Lambda(lambda x: (1. - K.filters.canny(x.unsqueeze(0))[0].clamp(0., 1.)).squeeze(0)),
        ])
        tf_inv = transforms.Compose([
            transforms.ToPILImage()
        ])
        return tf, tf_inv

class DONGJIN(VisionDataset):
    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - normal",
        "1 - anomaly",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.img_paths

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.img_paths

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_dir: str = 'train',
        test_dir: str = 'test'
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train # train or test
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_dir = os.path.join(self.raw_folder, f"{self.train_dir if self.train else self.test_dir}-images")
        self.label_file = os.path.join(self.raw_folder, f"{self.train_dir if self.train else self.test_dir}-labels")

        if self._check_legacy_exist():
            self.img_paths, self.targets = self._load_legacy_data()
            return

        if not self._check_exists():
            raise RuntimeError("Dataset not found at ", self.raw_folder)
        
        self.img_paths, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False
        return True
    
    def _load_legacy_data(self):
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))
    
    def _load_data(self):
        data = read_image_file(self.image_dir)
        targets = read_label_file(len(data))

        return data, targets

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        img_path, target = self.img_paths[index], int(self.targets[index])

        img = Image.open(os.path.join(self.image_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.img_paths)
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.isdir(os.path.join(self.raw_folder, f'{self.train_dir if self.train else self.test_dir}-images'))

def read_label_file(path: int) -> int:
    # TODO : dummy function
    return np.zeros((path,))
    
def read_image_file(path: str) -> Image:
    return [p for p in os.listdir(path) if p.endswith('.jpg')]
