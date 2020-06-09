"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib as plt
import cv2
from typing import Union
import torchvision.transforms as transforms

erosion = nn.MaxPool2d(5, stride=1, padding=2)

def normalize_0p1_to_n1p1(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize_n1p1_to_0p1(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list += [transforms.Normalize((-1.0,), (2.0,))]
    else:
        transform_list += [transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))]
    return transforms.Compose(transform_list)


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def calc_brightest_area(src, mask):
    erosion = nn.MaxPool2d(5, stride=1, padding=2)

    if torch.sum(mask) < 10:
        brightest_point = 1.0
    else:
        brightest_point = percentile(src[mask > 0.5], 80)

    brightest_mask = torch.zeros_like(mask)
    brightest_mask[src >= brightest_point] = 1.0
    brightest_mask = 1.0 - erosion(1.0-brightest_mask)
    brightest_mask = erosion(brightest_mask)
    brightest_mask = brightest_mask * mask
            
    if torch.sum(brightest_mask) < 10:
        max_value = torch.ones(1)
    else:
        max_value = torch.max(src[brightest_mask > 0.5])
    brightest = torch.zeros_like(brightest_mask)
    brightest = torch.clamp((src - brightest_point) / torch.clamp(max_value - brightest_point, min=1e-6), min=0)
    brightest = brightest * brightest_mask

    return brightest, brightest_point


# def tensor2im(input_image, imtype=np.uint8):
def tensor2im(input_image, imtype=np.uint8, gain=1.0, ch=0):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    # print('input_image.size:', input_image.size())
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[ch].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 * gain # post-processing: tranpose and scaling
        image_numpy = np.clip(image_numpy, 0.0, 255.0)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    h, w, c = image_numpy.shape
    if c==1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    image_pil = Image.fromarray(image_numpy)        
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
