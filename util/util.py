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
import torch.nn.functional as F
import kornia.filters
import random

random.seed(101)
# erosion = nn.MaxPool2d(15, stride=1, padding=7)
erosion = nn.MaxPool2d(11, stride=1, padding=5)

def min_loss_BC(pr_BC, gt_BC, bc_num, criterionBC):
    loss_G_BC = criterionBC(pr_BC, gt_BC[:, 0])
    for i in range(1, bc_num):
        loss_G_BC_cmp = criterionBC(pr_BC, gt_BC[:, i].squeeze(1))
        loss_G_BC = torch.min(loss_G_BC, loss_G_BC_cmp)
    return loss_G_BC

def min_loss_BC_NoBatch(pr_BC, gt_BC, bc_num, criterionBC):
    loss_G_BC = criterionBC(pr_BC, gt_BC[0])
    for i in range(1, bc_num):
        loss_G_BC_cmp = criterionBC(pr_BC, gt_BC[i])
        loss_G_BC = torch.min(loss_G_BC, loss_G_BC_cmp)
    return loss_G_BC


def get_current_pr_BC(pr_BC, mask, opt):
    pr_BP_BC = disp_brightest_coord(pr_BC, mask, opt.bp_tap, opt.bp_sigma)
    pr_BP_BC = (pr_BP_BC - 0.5) / 0.5
    return pr_BP_BC

def get_current_gt_BC(gt_BC, mask, opt):
    gt_BP_BC = disp_brightest_coord(gt_BC, mask, opt.bp_tap, opt.bp_sigma)
    gt_BP_BC = (gt_BP_BC - 0.5) / 0.5
    return gt_BP_BC

def get_current_BP(pr_BP, opt):
    pr_BP_norm = torch.squeeze(pr_BP, 0)*0.5+0.5
    mask_one = torch.ones_like(pr_BP_norm)
    _, _, pr_BP_BP, _ = calc_brightest(pr_BP_norm, mask_one, opt.bp_nr_tap, opt.bp_nr_sigma, opt.bp_tap, opt.bp_sigma)
    pr_BP_BP = (pr_BP_BP - 0.5) / 0.5
    pr_BP_BP = pr_BP_BP.unsqueeze(0)
    return pr_BP_BP

def calc_shading(img, albedo, mask, opt):
    img = torch.clamp(img * 0.5 + 0.5, min=0.0, max=1.0) # 0~1
    albedo = torch.clamp(albedo * 0.5 + 0.5, min=1e-6, max=1.0) # 0~1
    shading = img**2.2/albedo
    if opt.shading_norm:
        if torch.sum(mask) < 10:
            max_S = 1.0
        else:
            max_S = percentile(shading[mask.expand(shading.size()) > 0.5], 90)

        shading = shading/max_S

    shading = (shading - 0.5) / 0.5
    return torch.clamp(shading, min=-1.0, max=1.0) # -1~1


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

def mse(src, tar):
    diff2 = (torch.flatten(src) - torch.flatten(tar)) ** 2.0
    result = torch.sum(diff2) / (src.size(-2)*src.size(-1))
    return result

def mse_with_mask(src, tar, mask):
    diff2 = (torch.flatten(src) - torch.flatten(tar)) ** 2.0 * torch.flatten(mask > 0.5)
    if torch.sum(mask > 0.5) > 0:
        result = torch.sum(diff2) / torch.sum(mask > 0.5)
    else:
        result = torch.sum(mask > 0.5) # 0
    return result

def calc_dist(bc_gt, bc_tar):
    dist = 0.0
    cnt = 0
    for i in range(int(bc_gt[0][3])):
        for j in range(int(bc_tar[0][3])):
            dist_tmp = np.hypot(bc_gt[i][0] - bc_tar[j][0], bc_gt[i][1] - bc_tar[j][1])
            dist += dist_tmp
            cnt += 1
    return dist/float(cnt)


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


def calc_brightest(img, mask, nr_tap=11, nr_sigma=5.0, spread_tap=31, spread_sigma=5.0):
    # Blur the image (NR)
    img = torch.unsqueeze(img, 0) # To 4dim
#     gauss_nr = kornia.filters.GaussianBlur2d((nr_tap, nr_tap), (nr_sigma, nr_sigma))
#     img_blur = gauss_nr(img)
    median_nr = kornia.filters.MedianBlur((nr_tap, nr_tap))
    img_blur = median_nr(img)
    img_blur = torch.squeeze(img_blur, 0) # To 3dim
#     img_blur = img.clone()

    # Calc Brighest and top 20% values
    if torch.sum(mask[mask > 0.5]) < 10:
        brightest_max = torch.max(img_blur)
        brightest_20 = percentile(img_blur, 80)
    else:
        brightest_max = torch.max(img_blur[mask > 0.5]).item() # Get the brightest value (scalar)
#         brightest_max = torch.max(img_blur).item() # Get the brightest value (scalar)
        brightest_20 = percentile(img_blur[mask > 0.5], 80) # Get top 20% brighest value (scalar)
#         brightest_20 = percentile(img_blur, 80) # Get top 20% brighest value (scalar)
    
#     print('brightest_max:', brightest_max)
#     print('brightest_20:', brightest_20)

    # Calc 20% brightest area
#     brightest_area = torch.zeros_like(mask)
#     brightest_area = img_blur >= brightest_20
#     # brightest_area = brightest_area * (mask > 0.5)
#     brightest_area = img_blur * brightest_area
#     brightest_area = img.squeeze(0).clone()
    brightest_area = img_blur.clone()
    
    # Calc brightest pixel
    brightest_pixel = torch.zeros_like(brightest_area)
    brightest_pixel = img_blur >= brightest_max
#     brightest_pixel = brightest_pixel * (mask > 0.5)
    brightest_pixel_num = torch.sum(brightest_pixel).item()

    # Selects pixels to be picked up according to the number of brightest pixels
    if brightest_pixel_num < 1:
        print('Conditon 0: brightest_pixel_num:', brightest_pixel_num)
        brightest_coord = [(0.5, 0.5, 0, brightest_pixel_num)]
    elif brightest_pixel_num == 1:
        coord = torch.argmax(brightest_pixel.int())
        brightest_coord = (int(coord//brightest_pixel.size(2)), int(coord%brightest_pixel.size(2)))
        brightest_coord = [(float(brightest_coord[0])/float(brightest_pixel.size(1)), float(brightest_coord[1])/float(brightest_pixel.size(2)), 1, brightest_pixel_num)]
    elif brightest_pixel_num > 1:
        brightest_coord_list = torch.nonzero(brightest_pixel, as_tuple=False)
        brightest_coord = []
        for i in brightest_coord_list:
            coord = (float(i[1])/float(brightest_pixel.size(1)), float(i[2])/float(brightest_pixel.size(2)), 2, brightest_pixel_num)
            brightest_coord.append(coord)
#         pick_idx = random.randrange(0, len(brightest_coord_list))
#         print('Conditon 2: brightest_pixel_num:', brightest_pixel_num)
#         print('pick_idx:', pick_idx)
#         coord = brightest_coord_list[pick_idx]
#         brightest_coord = (float(coord[1])/float(brightest_pixel.size(1)), float(coord[2])/float(brightest_pixel.size(2)), 2, brightest_pixel_num)
    else:
        brightest_coord = [(0.5, 0.5, -1, brightest_pixel_num)]
        print('Conditon -1: Detected an exception.')

    # Spread the points in concentric circles
    brightest_pixel = torch.unsqueeze(brightest_pixel, 0) # To 4dim
    gauss_spread = kornia.filters.GaussianBlur2d((spread_tap, spread_tap), (spread_sigma, spread_sigma))
    brightest_pixel = gauss_spread(brightest_pixel.float())
    brightest_pixel = torch.squeeze(brightest_pixel, 0) # To 3dim
    

    # Normalize
#     brightest_area = torch.clamp((brightest_area - brightest_20) / torch.clamp(brightest_max - brightest_20, min=1e-6), min=0)
    clamp = lambda x: min(1.0, max(x, 1e-6))
    brightest_area = torch.clamp((brightest_area - brightest_20) / clamp(brightest_max - brightest_20), min=0, max=1)
    brightest_max_blur = torch.max(brightest_pixel)
    if brightest_max_blur > 0:
        brightest_pixel = brightest_pixel / brightest_max_blur

    return brightest_area, brightest_20, brightest_pixel, brightest_coord


def disp_brightest_coord(coord, mask, spread_tap=31, spread_sigma=5.0):
    brightest_pixel = torch.zeros_like(mask).float()
    h, w = brightest_pixel.size(-1), brightest_pixel.size(-2)
    coord_int = int(h*coord[0,0].item()) , int(w*coord[0,1].item())
    brightest_pixel[:, :, coord_int[0], coord_int[1]] = 1.0

    # Spread the points in concentric circles
    # brightest_pixel = torch.unsqueeze(brightest_pixel, 0) # To 4dim
    gauss_spread = kornia.filters.GaussianBlur2d((spread_tap, spread_tap), (spread_sigma, spread_sigma))
    brightest_pixel = gauss_spread(brightest_pixel.float())
    # brightest_pixel = torch.squeeze(brightest_pixel, 0) # To 3dim
    brightest_max_blur = torch.max(brightest_pixel)
    if brightest_max_blur > 0:
        brightest_pixel = brightest_pixel / brightest_max_blur
    return brightest_pixel

def disp_brightest_coords(coords, mask, spread_tap=31, spread_sigma=5.0):
    brightest_pixel = torch.zeros_like(mask).float()
    h, w = brightest_pixel.size(-1), brightest_pixel.size(-2)
    num = coord[0, 3]
    for i in range(num):
        coord_int = int(h*coord[i,0].item()) , int(w*coord[i,1].item())
        brightest_pixel[:, :, coord_int[0], coord_int[1]] = 1.0

    # Spread the points in concentric circles
    # brightest_pixel = torch.unsqueeze(brightest_pixel, 0) # To 4dim
    gauss_spread = kornia.filters.GaussianBlur2d((spread_tap, spread_tap), (spread_sigma, spread_sigma))
    brightest_pixel = gauss_spread(brightest_pixel.float())
    # brightest_pixel = torch.squeeze(brightest_pixel, 0) # To 3dim
    brightest_max_blur = torch.max(brightest_pixel)
    if brightest_max_blur > 0:
        brightest_pixel = brightest_pixel / brightest_max_blur
    return brightest_pixel


def im2tensor(input_numpy, grayscale=False):
    img = input_numpy.astype(np.float32)/255.0
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    image_tensor = transforms.Compose(transform_list)(img)
    return image_tensor

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
