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


class DatasetTrainVal(Dataset):

    def __init__(self, path, split, val_ratio, rotate=False, flip=False, grayscale=False, erase=0, resize=False, pad=False, preprocess=False):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = Path(path) / 'training' / 'images'
        gt_path = Path(path) / 'training' / 'groundtruth'

        # Listing the images and ground truth file paths
        self.images = [
            images_path / item
            for item in os.listdir(images_path)
            if item.endswith('.png')
        ]
        self.images.sort()

        self.gt = [
            gt_path / item
            for item in os.listdir(gt_path)
            if item.endswith('.png')
        ]
        self.gt.sort()

        # Divide to validation and training set based on the value of set_type
        idx = int(len(self.images) * val_ratio)
        if split == 'train':
            self.images = self.images[idx:]
            self.gt = self.gt[idx:]
        elif split == 'val':
            self.images = self.images[:idx]
            self.gt = self.gt[:idx]

        self.set_type = split
        self.rotate = rotate
        self.flip = flip
        self.grayscale = grayscale
        self.erase = erase
        self.resize = resize
        self.pad = pad
        self.preprocess = preprocess

    def transform(self, img, mask, index):
        """
        Augmenting the dataset by doing
            random horizontal flip
            random vertical flip 
            random rotations
            random erasing
            random grayscale
        """
        
        # Resize
        if self.resize:
            img = functional.resize(img, self.resize)
            mask = functional.resize(mask, self.resize)

        # Padding
        if self.pad:
            padd = transforms.Pad(padding=self.pad, padding_mode='reflect')
            img, mask = padd(img), padd(mask)
        
        # Do a vertical or horizontal flip randomly
        if self.flip and random.random() > 0.30:
            if random.random() > 0.5:
                img = functional.hflip(img)
                mask = functional.hflip(mask)
            else:
                img = functional.vflip(img)
                mask = functional.vflip(mask)

        # First apply a rotation based on diag_angles to extend dataset with non-horizontal and non-vertical roads and
        # then do a random rotate
        if self.rotate:
            diag_angles = [0, 15, 30, 45, 60, 75]
            img = functional.rotate(img, diag_angles[index % 6])
            mask = functional.rotate(mask, diag_angles[index % 6])
            angle = random.choice([0, 90, 180, 270])
            img = functional.rotate(img, angle)
            mask = functional.rotate(mask, angle)
            
        if self.grayscale and random.random() > 0.7:
            img = functional.rgb_to_grayscale(img, num_output_channels=3)

        # Transforming from PIL type to torch.tensor and normalizing the data to range [0, 1]
        to_tensor = transforms.ToTensor()
        img, mask = to_tensor(img), to_tensor(mask)
        
        if self.preprocess:
            params = smp.encoders.get_preprocessing_params("resnet50", "imagenet")
            img = (img - torch.tensor(params["mean"]).view(3, 1, 1)) / torch.tensor(params["std"]).view(3, 1, 1)
          
        # Erasing random rectangles from the image
        img = random_erase(img, n=self.erase, color_rgb='noise')

        return img, mask.round().long()

    def __getitem__(self, index):
        if self.rotate:
            # Getting each image 6 times, each time with a different diagonal rotation
            img, mask = self.images[index // 6], self.gt[index // 6]
        else:
            img, mask = self.images[index], self.gt[index]

        # Read image and ground truth files
        img = Image.open(img)
        mask = Image.open(mask)

        # Apply dataset augmentation transforms if needed
        img, mask = self.transform(img, mask, index)

        return img, mask

    def __len__(self):
        if self.rotate:
            return len(self.images) * 6
        return len(self.images)


class DatasetTest(Dataset):

    def __init__(self, path, preprocess=False):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = os.path.join(path, 'test_set_images')
        self.images = [
            os.path.join(images_path, item, item + '.png')
            for item in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, item))
        ]
        self.images.sort(key=lambda x: int(os.path.split(x)[-1][5:-4]))
        self.preprocess=preprocess

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img)

        # Transforming from PIL type to torch.tensor and normalizing the data to range [0, 1]
        img = transforms.ToTensor()(img)
        if self.preprocess:
            params = smp.encoders.get_preprocessing_params("resnet50", "imagenet")
            img = (img - torch.tensor(params["mean"]).view(3, 1, 1)) / torch.tensor(params["std"]).view(3, 1, 1)

        return img

    def __len__(self):
        return len(self.images)
