"""Oxford-IIIT Pet data module for three-class trimap segmentation."""
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
from torchvision.datasets import OxfordIIITPet


CLASS_NAMES = ["pet", "background", "border"]


class PetTrimapDataset(Dataset):
    """Wrap torchvision's OxfordIIITPet and apply paired transforms."""

    def __init__(self, root, split="trainval", indices=None, image_size=256,
                 augmentation=False, download=True):
        self.root = str(Path(root).expanduser())
        self.image_size = int(image_size)
        self.augmentation = augmentation
        self.dataset = OxfordIIITPet(
            root=self.root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.indices = list(range(len(self.dataset))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def _resize(self, image, mask):
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        return image, mask

    def __getitem__(self, item):
        image, mask = self.dataset[self.indices[item]]
        image, mask = self._resize(image.convert("RGB"), mask)

        # Geometric augmentation is applied identically to image and mask.
        if self.augmentation:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.25:
                angle = random.uniform(-12, 12)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=2)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        # ImageNet normalization for the pretrained ResNet encoder.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        image_tensor = (image_tensor - mean) / std

        # Official trimap values: 1=pet, 2=background, 3=border.
        # Convert to contiguous class IDs 0, 1, 2 for CrossEntropyLoss.
        mask_array = np.asarray(mask, dtype=np.uint8).astype(np.int64) - 1
        mask_tensor = torch.from_numpy(mask_array.copy()).long()
        if mask_tensor.min() < 0 or mask_tensor.max() > 2:
            raise ValueError("Unexpected trimap values; expected original mask labels 1, 2, 3.")
        return image_tensor, mask_tensor


class OxfordPetDataModule(pl.LightningDataModule):
    def __init__(self, root="./data", image_size=256, batch_size=8,
                 num_workers=2, val_fraction=0.2, seed=42,
                 augmentation=False, download=True, limit_train=0, limit_val=0):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.augmentation = augmentation
        self.download = download
        self.limit_train = limit_train
        self.limit_val = limit_val

    def prepare_data(self):
        # Download once; torchvision handles already-downloaded files.
        OxfordIIITPet(root=str(Path(self.root).expanduser()), split="trainval",
                      target_types="segmentation", download=self.download)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            base = OxfordIIITPet(
                root=str(Path(self.root).expanduser()), split="trainval",
                target_types="segmentation", download=self.download,
            )
            n_total = len(base)
            n_val = max(1, int(round(n_total * self.val_fraction)))
            n_train = n_total - n_val
            generator = torch.Generator().manual_seed(self.seed)
            train_subset, val_subset = random_split(base, [n_train, n_val], generator=generator)
            train_indices = train_subset.indices
            val_indices = val_subset.indices

            if self.limit_train > 0:
                train_indices = train_indices[:self.limit_train]
            if self.limit_val > 0:
                val_indices = val_indices[:self.limit_val]

            self.train_dataset = PetTrimapDataset(
                self.root, "trainval", train_indices, self.image_size,
                augmentation=self.augmentation, download=self.download,
            )
            self.val_dataset = PetTrimapDataset(
                self.root, "trainval", val_indices, self.image_size,
                augmentation=False, download=self.download,
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = PetTrimapDataset(
                self.root, "test", None, self.image_size,
                augmentation=False, download=self.download,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.num_workers > 0)
