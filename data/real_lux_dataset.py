from data.base_dataset import BaseDataset, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import cv2
import os
import torch
from util import util

class RealLuxDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        print(opt.dataroot+'_gtlx')
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        # self.gtlx_paths = sorted(make_dataset(opt.dataroot+'_gtlx', opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform_lux = get_transform(opt, grayscale=True, convert=False, method=Image.BILINEAR)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        gtlx_path = os.path.splitext(A_path)[0] + '_lux.csv'
        print('img:', A_path)
        print('gxlx:', gtlx_path)
        srgb_img = Image.open(A_path).convert('RGB')
        gtlx_np = np.loadtxt(gtlx_path, delimiter=',')
        gtlx_pil = Image.fromarray(gtlx_np)
        srgb_img = self.transform(srgb_img)
        gt_SH = self.transform_lux(gtlx_pil)
        gt_SH = gt_SH / gt_SH.max()
        mask = torch.ones_like(gt_SH)

        gt_BA, brightest_20, gt_BP, gt_BC\
             = util.calc_brightest(
                 gt_SH, mask,
                 nr_tap=self.opt.bp_nr_tap, 
                 nr_sigma=self.opt.bp_nr_sigma,
                 spread_tap=self.opt.bp_tap, 
                 spread_sigma=self.opt.bp_sigma
                 )

        gt_SH = normalize(grayscale=True)(gt_SH)
        mask = normalize(grayscale=True)(mask)
        gt_BA = normalize(grayscale=True)(gt_BA)
        gt_BP = normalize(grayscale=True)(gt_BP)
        gt_BC = torch.Tensor(list(gt_BC))

        srgb_img = torch.unsqueeze(srgb_img, 0) # [1, 3, 256, 256]
        gt_AL = torch.zeros_like(srgb_img)
        gt_SH = torch.unsqueeze(gt_SH, 0)
        mask = torch.unsqueeze(mask, 0)
        gt_BA = torch.unsqueeze(gt_BA, 0)
        gt_BP = torch.unsqueeze(gt_BP, 0)

        return {'A': srgb_img, 'gt_AL': gt_AL, 'gt_SH': gt_SH, 'mask': mask, 'gt_BA': gt_BA, 'gt_BP': gt_BP, 'gt_BC':gt_BC, 'A_paths': A_path}
        # return {'A': A, 'gtlx':gtlx, 'A_paths': A_path, 'gtlx_paths': gtlx_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
