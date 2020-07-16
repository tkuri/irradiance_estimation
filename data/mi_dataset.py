import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pickle
import torch
import torch.nn as nn
import skimage
from skimage.restoration import denoise_tv_chambolle
from util import util

# def make_dataset(list_dir, max_dataset_size=float("inf"), phase='train'):
#     file_name = list_dir + "img_batch_{}.p".format(phase)
#     print('load image list:', file_name)
#     images_list = pickle.load( open( file_name, "rb" ) )

#     return images_list[:min(max_dataset_size, len(images_list))]


# def MI_make_dataset(list_dir):
#     file_name = list_dir + "list_sub4.txt"
#     dir_list = pickle.load( open( file_name, "rb" ) )

#     images_list = []
#     lights_list = []
#     for d in dir_list:
#         images_list += [d + '/dir_{}_mip2_input.jpg'.format(i) for i in range(25)]
#         lights_list += [d + '/dir_{}_mip2_L.jpg'.format(i) for i in range(25)]
#         SHs_list += [d + '/dir_{}_mip2_SH.jpg'.format(i) for i in range(25)]

#     print('images_list:', images_list)
#     return images_list, lights_list, SHs_list

def MI_make_dataset(list_dir):
    file_name = list_dir + "list_sub4.txt"
    dir_list = pickle.load( open( file_name, "rb" ) )

    print('images_list:', dir_list)
    return dir_list


class MiDataset(BaseDataset):
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
        self.dataroot = opt.dataroot 
        list_dir = self.dataroot + '/train_list/'
        self.img_dirs = MI_make_dataset(list_dir)
        if len(self.img_dirs) == 0:
            raise(RuntimeError("Found 0 directories in directory \n"))
        assert(self.opt.load_size >= self.opt.crop_size)

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
        srgb_img = []
        L = []
        gt_SH = []
        path = self.img_dirs[index]
        for i in range(25):
            img_path = self.dataroot + '/results/' + path + '/dir_{}_mip2_input.jpg'.format(i)
            srgb_img.append(Image.open(img_path).convert('RGB'))

            L_path = self.dataroot + '/results/' + path + '/dir_{}_mip2_L.jpg'.format(i)
            L.append(Image.open(L_path).convert('RGB'))

            SH_path = self.dataroot + '/results/' + path + '/dir_{}_mip2_SH.jpg'.format(i)
            gt_SH.append(Image.open(SH_path).convert('RGB'))


        # apply the same transform to both A and B
        transform_params = get_params(self.opt, srgb_img.size)
        srgb_img_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        L_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        SH_transform = get_transform(self.opt, transform_params, grayscale=True, convert=False)

        for i in range(25):
            srgb_img[i] = srgb_img_transform(srgb_img[i])
            gt_SH[i] = SH_transform(gt_SH[i])
            L[i] = L_transform(L[i])

        mask = torch.ones_like(L[0])
        result = {}
        res = []
        for i in range(25):
            res_tmp = make_bp_data(srgb_img[i], gt_SH[i], mask, self.opt)
            res.append(res_tmp)

        srgb_img_cat = torch.cat([res[i]['A'] for i in range(25)], dim=0)
        gt_SH_cat = torch.cat([res[i]['gt_SH'] for i in range(25)], dim=0)
        gt_BA_cat = torch.cat([res[i]['gt_BA'] for i in range(25)], dim=0)
        # gt_BP_cat = torch.cat([res[i]['gt_BP'] for i in range(25)], dim=0)
        # gt_BC_cat = torch.cat([res[i]['gt_BC'] for i in range(25)], dim=0)
        L_cat = torch.cat([torch.unsqueeze(L[i], 0) for i in range(25)], dim=0)
        
        result['A'] = srgb_img_cat
        result['gt_SH'] = gt_SH_cat
        result['L'] = L_cat
        result['mask'] = torch.unsqueeze(mask, 0)

        result['gt_BA'] = gt_BA_cat
        # result['gt_BP'] = gt_BP_cat
        # result['gt_BC'] = gt_BC_cat
        result['gt_BC'] = []
        for i in range(25):
            result['gt_BC'].append(res[i]['gt_BC'])
        
        result['A_paths'] = path

        return {'A': srgb_img, 'gt_AL': gt_AL, 'gt_SH': gt_SH, 'mask': mask, 'gt_BA': gt_BA, 'gt_BP': gt_BP, 'gt_BC':gt_BC, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.img_dirs)