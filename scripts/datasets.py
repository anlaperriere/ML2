import os
import random
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as functional
from PIL import Image
from pathlib import Path
from helpers import random_erase

"""
Script to create training, validation and test datasets and to perform data augmentation.
Classes inheriting from the torch Dataset class are created.
"""


class DatasetTrainVal(Dataset):
    def __init__(
        self,
        path,
        split,
        val_ratio,
        rotate=False,
        flip=False,
        grayscale=False,
        erase=0,
        resize=False,
    ):
        super(Dataset, self).__init__()

        # Paths to satellite image and ground truth obtention
        images_path = Path(path) / "training" / "images"
        gt_path = Path(path) / "training" / "groundtruth"

        # File paths listing and sorting
        self.images = [
            images_path / item
            for item in os.listdir(images_path)
            if item.endswith(".png")
        ]
        self.images.sort()
        self.gt = [
            gt_path / item for item in os.listdir(gt_path) if item.endswith(".png")
        ]
        self.gt.sort()

        # Division of the dataset into training and validation sets based on split and validation ratio
        idx = int(len(self.images) * val_ratio)
        if split == "train":
            self.images = self.images[idx:]
            self.gt = self.gt[idx:]
        elif split == "val":
            self.images = self.images[:idx]
            self.gt = self.gt[:idx]

        self.set_type = split
        self.rotate = rotate
        self.flip = flip
        self.grayscale = grayscale
        self.erase = erase
        self.resize = resize

    def transform(self, img, mask, index):
        """
        To augment the dataset by doing
            random horizontal flip
            random vertical flip
            random rotations
            random grayscaling
            random erasing
        """

        # Resizing if neeeded
        if self.resize:
            img = functional.resize(img, self.resize)
            mask = functional.resize(mask, self.resize)

        # Random vertical or horizontal flips
        if self.flip and random.random() > 0.30:
            if random.random() > 0.5:
                img = functional.hflip(img)
                mask = functional.hflip(mask)
            else:
                img = functional.vflip(img)
                mask = functional.vflip(mask)

        # 6 rotations per image based on diagonal angles followed by a random rotation
        if self.rotate:
            diag_angles = [0, 15, 30, 45, 60, 75]
            img = functional.rotate(img, diag_angles[index % 6])
            mask = functional.rotate(mask, diag_angles[index % 6])
            angle = random.choice([0, 90, 180, 270])
            img = functional.rotate(img, angle)
            mask = functional.rotate(mask, angle)

        # Random grayscaling
        if self.grayscale and random.random() > 0.7:
            img = functional.rgb_to_grayscale(img, num_output_channels=3)

        # PIL image conversion to torch tensor in the range [0., 1.]
        to_tensor = transforms.ToTensor()
        img, mask = to_tensor(img), to_tensor(mask)

        # Random rectangles erasing
        img = random_erase(img, n=self.erase, color_rgb="noise")

        return img, mask.round()

    def __getitem__(self, index):
        if self.rotate:
            # Each image is loaded 6 times to then perform 6 different diagonal angles rotations
            img, mask = self.images[index // 6], self.gt[index // 6]
        else:
            img, mask = self.images[index], self.gt[index]

        # Satellite image and ground truth reading using PIL
        img = Image.open(img)
        mask = Image.open(mask)

        # Data augmentation
        img, mask = self.transform(img, mask, index)

        return img, mask

    def __len__(self):
        if self.rotate:
            # Each image has been loaded 6 times
            return len(self.images) * 6
        return len(self.images)


class DatasetTest(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()

        # Path to satellite image obtention
        images_path = os.path.join(path, "test_set_images")

        # File paths listing and sorting
        self.images = [
            os.path.join(images_path, item, item + ".png")
            for item in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, item))
        ]
        self.images.sort(key=lambda x: int(os.path.split(x)[-1][5:-4]))

    def __getitem__(self, index):
        img = self.images[index]

        # Satellite image reading using PIL
        img = Image.open(img)

        # PIL image conversion to torch tensor in the range [0., 1.]
        img = transforms.ToTensor()(img)

        return img

    def __len__(self):
        return len(self.images)
