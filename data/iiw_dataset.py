import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pickle
import torch
import torch.nn as nn
import skimage
from typing import Union
import numpy as np
import random


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

def make_dataset(list_dir, max_dataset_size=float("inf"), iiw=False):
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )

    if iiw:
        concat_list = images_list[0]+images_list[1]+images_list[2]
        dataset = concat_list[:min(max_dataset_size, len(concat_list))]
    else:    
        dataset = images_list[:min(max_dataset_size, len(images_list))]

    return dataset

class IIWDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot # ../CGIntrinsics/CGIntrinsics/
        self.img_paths_iiw = make_dataset(self.dataroot + '/IIW/train_list/', opt.max_dataset_size, iiw=True)
        if llen(self.img_paths_iiw) == 0:
            raise(RuntimeError("Found 0 images in: " + self.dataroot + '/IIW/train_list/'+ "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        self.A_size = len(self.img_paths_iiw)  # get the size of dataset B
        input_nc = self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.output_nc      # get the number of channels of output image
        self.transform = get_transform(self.opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # read a image given a random integer index
        if self.opt.serial_batches:   # make sure index is within then range
            index_A = index % self.A_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_A = random.randint(0, self.A_size - 1)
        img_path_iiw = self.img_paths_iiw[index_A]
        img_path_iiw = self.dataroot + "/IIW/iiw-dataset/data/" + img_path_iiw.split('/')[-1][:-3]
        img_iiw = Image.open(img_path_iiw).convert('RGB')
        
        # apply the same transform to both A and B
        img_iiw = self.transform(img_iiw)

        # img_cg = torch.unsqueeze(img_cg, 0) # [1, 3, 256, 256]
        # img_cg = torch.unsqueeze(img_cg, 0) # [1, 3, 256, 256]
        
        return {'A': img_iiw, 'A_paths': img_path_iiw}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.img_paths_iiw)