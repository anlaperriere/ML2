import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_features(img):
    """
    Extract 6-dimensional features consisting of average RGB color as well as variance
    """
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_2d(img):
    """
    Extract 2-dimensional features consisting of average gray color as well as variance
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


def extract_img_features(filename):
    """
    Extract features for a given image
    """
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X


def load_training_data(folder_dir, n=100):
    """
    Loads all the training files
    """
    img_dir = folder_dir + "images/"
    files = os.listdir(img_dir)
    n = min(n, len(files))  
    print("Loading " + str(n) + " images")
    imgs = [load_image(img_dir + files[i]) for i in range(n)]

    gnd_dir = folder_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gnd_imgs = [load_image(gnd_dir + files[i]) for i in range(n)]
    return imgs, gnd_imgs