"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import pickle
import numpy as np
import imageio as io
from skimage.transform import resize
import math, random
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 2] = s / 3.0
    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    return irg
    
def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg


def make_dataset_iiw(list_dir):
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )

    return images_list

class IIW_ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, mode, is_flip, transform=None, 
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset_iiw(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.is_flip = is_flip    
        self.img_list = img_list
        # self.targets_list = targets_list
        # self.img_list_2 = img_list_2
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.half_window = 1
        self.current_o_idx = mode
        self.set_o_idx(mode)
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)

    def set_o_idx(self, o_idx):
        self.current_o_idx = o_idx

        if o_idx == 0:
            self.height = 256
            self.width = 384
        elif o_idx == 1:
            self.height = 384
            self.width = 256
        elif o_idx == 2:
            self.height = 384
            self.width = 384
        elif o_idx == 3:
            self.height = 384
            self.width = 512
        else:
            self.height = 512
            self.width = 384

    def DA(self, img, mode, random_filp):

        # if random_filp > 0.5:
            # img = np.fliplr(img)

        # img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]

        img = resize(img, (self.height, self.width), order = mode)

        return img

    # def iiw_loader(self, img_path):
    #     # img = np.float32(io.imread(img_path))/ 255.0

    #     hdf5_file_read_img = h5py.File(img_path,'r')
    #     img = hdf5_file_read_img.get('/iiw/img')
    #     img = np.float32(np.array(img))

    #     img = np.transpose(img, (2,1, 0))
    #     hdf5_file_read_img.close()

    #     random_filp = random.random()

    #     if self.is_flip and random_filp > 0.5:
    #         img = np.fliplr(img)

    #     return img, random_filp

    def iiw_loader(self, img_path):
        
        img_path = img_path[-1][:-3]
        img_path = self.root + "/IIW/iiw-dataset/data/" + img_path
        img = np.float32(io.imread(img_path))/ 255.0
        oringinal_shape = img.shape

        img = resize(img, (self.height, self.width))

        random_filp = random.random()

        if self.is_flip and random_filp > 0.5:
            img = np.fliplr(img)

        return img, random_filp, oringinal_shape

    def construst_R_weights(self, N_feature):

        center_feature = np.repeat( np.expand_dims(N_feature[4, :, :,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - N_feature

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:2]**2  , 3) / (self.sigma_chro**2)) \
                    * np.exp(- (feature_diff[:,:,:,2]**2) /(self.sigma_I**2) )

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :] 
                ct_idx += 1

        return sub_C


    def __getitem__(self, index):

        targets_1 = {}
        # temp_targets = {}

        # img_path = self.root + "/CGIntrinsics/IIW/" + self.img_list[self.current_o_idx][index]
        # judgement_path = self.root + "/CGIntrinsics/IIW/data/" + img_path.split('/')[-1][0:-6] + 'json'
        # mat_path = self.root + "/CGIntrinsics/IIW/long_range_data_4/" + img_path.split('/')[-1][0:-6] + "h5"
        img_path = self.root + "/IIW/" + self.img_list[self.current_o_idx][index]
        judgement_path = self.root + "/IIW/iiw-dataset/data/" + img_path.split('/')[-1][0:-6] + 'json'
        mat_path = self.root + "/IIW/long_range_data_4/" + img_path.split('/')[-1][0:-6] + "h5"
        targets_1['mat_path'] = mat_path

        # img, random_filp = self.iiw_loader(img_path)
        srgb_img, random_filp, oringinal_shape = self.iiw_loader(self.img_list[self.current_o_idx][index].split('/'))

        targets_1['path'] = "/" + img_path.split('/')[-1] 
        targets_1["judgements_path"] = judgement_path
        targets_1["random_filp"] = random_filp > 0.5
        targets_1["oringinal_shape"] = oringinal_shape

        # if random_filp > 0.5:
            # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch_flip/" + img_path.split('/')[-1] + "/R0.h5"
        # else:
            # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch/" + img_path.split('/')[-1] + "/R0.h5"

        rgb_img = srgb_to_rgb(srgb_img)
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()
        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()

        for i in range(0, self.num_scale):
            feature_3d = rgb_to_irg(rgb_img)
            sub_matrix = self.construst_sub_matrix(feature_3d)
            r_w = self.construst_R_weights(sub_matrix)
            targets_1['r_w_s'+ str(i)] = torch.from_numpy(r_w).float()
            rgb_img = rgb_img[::2,::2,:]



        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(srgb_img, (2,0,1)))).contiguous().float()

        sparse_shading_name = str(self.height) + "x" + str(self.width)

        if self.current_o_idx == 0:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 1:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name +  "/R0.h5"
        elif self.current_o_idx == 2:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 3:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 4:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"

        return final_img, targets_1, sparse_path_1s


    def __len__(self):
        return len(self.img_list[self.current_o_idx])