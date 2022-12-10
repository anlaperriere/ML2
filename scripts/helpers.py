import math
import os
import re
import torch
import torch.nn.functional as functional
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import PIL.Image as Image
from sklearn.metrics import f1_score
import random


# Model loading and saving
def load_model(model, opti, args):
    """
    To load pretrained model weights and optimizer states
    """
    if args.device == "cuda" and torch.cuda.is_available():
        checkpoint = torch.load(args.weight_path, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        opti.load_state_dict(checkpoint['optimizer_state_dict'])
    elif args.device == "mps" and torch.backends.mps.is_available():
        checkpoint = torch.load(args.weight_path, map_location=torch.device('mps'))
        model.load_state_dict(checkpoint['model_state_dict'])
        opti.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint = torch.load(args.weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        opti.load_state_dict(checkpoint['optimizer_state_dict'])


def save_model(model, opti, path, args):
    """
    To save trained model and optimizer states
    """

    save_path = os.path.join(path, args.experiment_name + '.pt')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opti.state_dict(),
    }, save_path)


# Loss computation
# Dice loss derived from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
def dice_loss(output, mask, smooth=1.0):
    """
    To compute the dice loss
    """

    output = output[:, 0].contiguous().view(-1)
    mask = mask[:, 0].contiguous().view(-1)
    intersection = (output * mask).sum()
    dsc = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)
    return 1. - dsc


def create_folder(path):
    """
    To create a new folder
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(output, idx, path, threshold=0.5):
    """
    Thresholds the output and ...TODO...
        ouput : torch tensor, with values ranging from 0 to 1 (sigmoid probabilities)
    
    """
    labels = (output > threshold).squeeze().cpu().numpy()
    img = Image.fromarray((labels * 255).astype(np.uint8))
    img.save(os.path.join(path, 'satImage_{:03d}.png'.format(idx)))


def save_image_overlap(output, img, idx, path):
    """
    Highlights the predicted roads on the image and saves it.
    """
    labels = (output > 0.5).cpu().numpy().squeeze()
    img = img.cpu().numpy().squeeze() * 255
    img[2, labels] = 150
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img = Image.fromarray(img.astype(np.uint8), 'RGB')
    img.save(os.path.join(path, 'satImage_overlap_{:03d}.png'.format(idx)))


foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def mask_to_patches(im):
    """
    Convert an image into the patches used by the submission format.
    """
    patch_size = 16
    x = len(im.shape) - 2
    y = len(im.shape) - 1
    # pad image
    out0 = math.ceil(im.shape[x] / patch_size)
    out1 = math.ceil(im.shape[y] / patch_size)
    pad0 = out0 - im.shape[x] // patch_size
    pad1 = out1 - im.shape[y] // patch_size
    padded = functional.pad(im, (0, 0, 0, 0, 0, pad0, 0, pad1), value=foreground_threshold)
    # convert to patches
    patches = padded.unfold(x, patch_size, patch_size).unfold(y, patch_size, patch_size)
    # apply threshold
    return torch.mean(patches.float(), dim=(x + 2, y + 2))


def random_erase(img, n=1, scale=(0, 0.1), rgb=(.5, .5, .5)):
    """
    Erase random rectangles from the image.
        img: torch.tensor
        n: number of rectangles
        scale: range of width and height with respect to the size of the image
        rgb: color of the rectangle
    """

    for _ in range(n):
        c = (random.randint(0, img.shape[1]), random.randint(0, img.shape[2]))
        h = round(img.shape[1] * (scale[0] + random.random() * scale[1]))
        w = round(img.shape[2] * (scale[0] + random.random() * scale[1]))
        if rgb == 'noise':
            for i in range(c[0] - h, c[0]):
                for j in range(c[1] - w, c[1]):
                    img[0, i, j] = random.random()
                    img[1, i, j] = random.random()
                    img[2, i, j] = random.random()
        else:
            img[0, c[0] - h:c[0], c[1] - w:c[1]] = rgb[0]
            img[1, c[0] - h:c[0], c[1] - w:c[1]] = rgb[1]
            img[2, c[0] - h:c[0], c[1] - w:c[1]] = rgb[2]
    return img


def get_score_patches(output, mask, threshold=foreground_threshold):
    """
    Calculate F1 score of the prediction, grouping by patches first.
    """
    output_patches = mask_to_patches(output)
    mask_patches = mask_to_patches(mask)
    output_labels = output_patches > threshold
    mask_labels = mask_patches > threshold
    mask_ = np.reshape(mask_labels.cpu().numpy(), (mask.shape[0], -1))
    labels_ = np.reshape(output_labels.cpu().numpy(), (output.shape[0], -1))
    # Calculating f1_score
    f_score = f1_score(mask_, labels_, average='macro', zero_division=0)

    return f_score


def get_score(output, mask, threshold=0.5):
    """
    Calculate F1 score of the prediction
    """
    labels_ = output > threshold
    mask_ = np.reshape(mask.cpu().numpy(), (mask.shape[0], -1))
    labels_ = np.reshape(labels_.cpu().numpy(), (labels_.shape[0], -1))
    # Calculating f1_score
    f_score = f1_score(mask_, labels_, average='macro', zero_division=0)

    return f_score


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """
    Reads a single image and outputs the strings that should go into the submission file
    """
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)


def masks_to_submission(submission_filename, *image_filenames):
    """
    Converts images into a submission file
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


cols = {'train': {'loss': [], 'f1-score': [], 'f1_patch': []}, 
        'val': {'loss': [], 'f1-score': [], 'f1_patch': []}}


def save_track(path, args, train_loss=None, train_f1=None, train_f1_patch=None, val_loss=None, val_f1=None,
               val_f1_patch=None):
    """
    Saves the result of the epoch in the corresponding file.
    """
    if train_loss is not None:
        cols['train']['loss'].append(train_loss)
    if train_f1 is not None:
        cols['train']['f1-score'].append(train_f1)
    if train_f1_patch is not None:
        cols['train']['f1_patch'].append(train_f1_patch)

    if val_loss is not None:
        cols['val']['loss'].append(val_loss)
    if val_f1 is not None:
        cols['val']['f1-score'].append(val_f1)
    if val_f1_patch is not None:
        cols['val']['f1_patch'].append(val_f1_patch)

    df = pd.DataFrame.from_dict(cols['train'])
    df.to_csv(os.path.join(path, args.experiment_name + "_train_tracking.csv"))

    df = pd.DataFrame.from_dict(cols['val'])
    df.to_csv(os.path.join(path, args.experiment_name + "_val_tracking.csv"))    