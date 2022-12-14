import math
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import PIL.Image as Image
import torch
import torch.nn.functional as functional
from sklearn.metrics import f1_score


# ------------------------- Tools for image modifications -------------------------

def random_erase(img, n=1, s_range=(0, 0.1), color_rgb=(.5, .5, .5)):
    """
    Inputs:
        img (torch tensor): image on which rectangles will be erased
        n (int): number of rectangles
        s_range (float tuple): to generate random rectangle shapes
        color_rgb (float tuple or string): to fill the erased rectangle
    Output:
        Image with erased rectangle
    """
    for rec in range(n):

        # Location of the erased rectangle
        loc = (random.randint(0, img.shape[1]), random.randint(0, img.shape[2]))

        # Generating random rectangle shape
        h = round(img.shape[1] * (s_range[0] + random.random() * s_range[1]))
        w = round(img.shape[2] * (s_range[0] + random.random() * s_range[1]))

        if color_rgb == 'noise':
            # Fill with random noise
            for i in range(loc[0] - h, loc[0]):
                for j in range(loc[1] - w, loc[1]):
                    img[0, i, j] = random.random()
                    img[1, i, j] = random.random()
                    img[2, i, j] = random.random()
        else:
            # Fill with color
            img[0, loc[0] - h:loc[0], loc[1] - w:loc[1]] = color_rgb[0]
            img[1, loc[0] - h:loc[0], loc[1] - w:loc[1]] = color_rgb[1]
            img[2, loc[0] - h:loc[0], loc[1] - w:loc[1]] = color_rgb[2]

    return img


# ------------------------- Tools for weights loading and saving -------------------------

def load_model(model, optimizer, device, weights_path):
    """
    To load pretrained model weights and optimizer states
    """
    if device == "cuda" and torch.cuda.is_available():
        checkpoint = torch.load(weights_path, map_location=torch.device('cuda'))
    elif device == "mps" and torch.backends.mps.is_available():
        checkpoint = torch.load(weights_path, map_location=torch.device('mps'))
    else:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_model(model, optimizer, path, experiment):
    """
    To save trained model weights and optimizer states
    """
    save_path = os.path.join(path, experiment + '.pt')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)


# ------------------------- Tools for dice loss computation -------------------------

def dice_loss(output, mask, smooth=1.0):
    # Dice loss derived from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
    """
    Inputs:
        output (torch tensor): model prediction
        mask (torch tensor): ground truth
        smooth (float): to avoid zero-division
    Output:
        Computed dice loss
    """
    output = output[:, 0].contiguous().view(-1)
    mask = mask[:, 0].contiguous().view(-1)
    intersection = (output * mask).sum()
    # Dice similarity coefficient
    dsc = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)
    return 1. - dsc


# ------------------------- Tools for results saving -------------------------

def create_folder(path):
    """
    To create a new folder located at a specified path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print("Folder created: {}".format(str(path)))


def save_track(path, experiment, train_loss=None, train_f1=None, train_f1_patch=None,
               val_loss=None, val_f1=None, val_f1_patch=None):
    """
    To save the loss and f1-scores at current epoch in a csv file
    """
    # Train and validation csv headers
    cols = {'train': {'loss': [], 'f1-score': [], 'f1_patch': []},
            'val': {'loss': [], 'f1-score': [], 'f1_patch': []}}

    if train_loss:
        cols['train']['loss'].append(train_loss)
    if train_f1:
        cols['train']['f1-score'].append(train_f1)
    if train_f1_patch:
        cols['train']['f1_patch'].append(train_f1_patch)

    df = pd.DataFrame.from_dict(cols['train'])
    df.to_csv(os.path.join(path, experiment + "_train_tracking.csv"))

    if val_loss:
        cols['val']['loss'].append(val_loss)
    if val_f1:
        cols['val']['f1-score'].append(val_f1)
    if val_f1_patch:
        cols['val']['f1_patch'].append(val_f1_patch)

    df = pd.DataFrame.from_dict(cols['val'])
    df.to_csv(os.path.join(path, experiment + "_val_tracking.csv"))


proba_threshold = 0.5  # Sigmoid probabilities range from 0 to 1, the threshold background vs foreground is set at 0.5


def save_image(output, idx, path, threshold=proba_threshold):
    """
    To save a binary mask image obtained after thresholding the model prediction on image idx
    """
    # Thresholds the output probabilities
    labels = (output > threshold).squeeze().cpu().numpy()
    # Convert to image
    img = Image.fromarray((labels * 255).astype(np.uint8))
    # Save image
    img.save(os.path.join(path, 'satImage_{:03d}.png'.format(idx)))


def save_image_overlap(output, img, idx, path, threshold=proba_threshold):
    """
    To save the image idx with highlighted predicted roads
    """
    # Thresholds the output probabilities
    labels = (output > threshold).cpu().numpy().squeeze()
    # Overlap image with road prediction
    img = img.cpu().numpy().squeeze() * 255
    img[2, labels] = 150
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img = Image.fromarray(img.astype(np.uint8), 'RGB')
    # Save image
    img.save(os.path.join(path, 'satImage_overlap_{:03d}.png'.format(idx)))


# ------------------------- Tools for patch-wise and pixel-wise evaluation -------------------------


def get_score(output, mask, threshold=proba_threshold):
    """
    Inputs:
        output (torch tensor): model prediction
        mask (torch tensor): ground truth
        threshold (float): the probability threshold to label a pixel as background or foreground
    Output:
        Computed pixel-wise f1-score
    """
    # Threshold pixels
    labels = output > threshold
    # Reshape
    mask = np.reshape(mask.cpu().numpy(), (mask.shape[0], -1))
    labels = np.reshape(labels.cpu().numpy(), (labels.shape[0], -1))
    # Compute f1-score
    f_score = f1_score(mask, labels, average='macro', zero_division=0)

    return f_score


# Percentage of foreground pixels in a patch required to assign a foreground label to a patch
foreground_threshold = 0.25


def mask_to_patches(img, threshold=foreground_threshold):
    """
    To convert an image into 16x16 patches as the submission format
    """
    patch_size = 16
    # rgb images: len(shape)=3
    x = len(img.shape) - 2
    y = len(img.shape) - 1
    # Image padding
    out0 = math.ceil(img.shape[x] / patch_size)
    out1 = math.ceil(img.shape[y] / patch_size)
    pad0 = out0 - img.shape[x] // patch_size
    pad1 = out1 - img.shape[y] // patch_size
    padded = functional.pad(img, (0, 0, 0, 0, 0, pad0, 0, pad1), value=threshold)
    # Create patches in image
    patches = padded.unfold(x, patch_size, patch_size).unfold(y, patch_size, patch_size)
    return torch.mean(patches.float(), dim=(x + 2, y + 2))


def get_score_patches(output, mask, threshold=foreground_threshold):
    """
    Inputs:
        output (torch tensor): model prediction
        mask (torch tensor): ground truth
        threshold (float): the probability threshold to label a pixel as background or foreground
    Output:
        Computed patch-wise f1-score
    """
    # Convert to patches
    output_patches = mask_to_patches(output)
    mask_patches = mask_to_patches(mask)
    # Apply threshold
    output_labels = output_patches > threshold
    mask_labels = mask_patches > threshold
    # Reshape
    mask_ = np.reshape(mask_labels.cpu().numpy(), (mask.shape[0], -1))
    labels_ = np.reshape(output_labels.cpu().numpy(), (output.shape[0], -1))
    # Compute f1_score
    f_score = f1_score(mask_, labels_, average='macro', zero_division=0)

    return f_score


# ------------------------- Tools for submission formatting -------------------------


def patch_to_label(patch, threshold=foreground_threshold):
    """
    Returns 1 if the patch is assigned to foreground (road) and 0 otherwise
    """
    m = np.mean(patch)
    if m > threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """
    From a test image name, outputs the patches names and their predictions for the submission format
    """
    # Finds the name of the test image
    img_number = int(re.search(r"\d+", image_filename).group(0))
    # Reads the image
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            # Cut image in patches
            patch = im[i:i + patch_size, j:j + patch_size]
            # Prediction label for each patch
            label = patch_to_label(patch)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)


def masks_to_submission(submission_filename, *image_filenames):
    """
    To convert images into a submission csv file
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
