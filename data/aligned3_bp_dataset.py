import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import torch
from util import util

class Aligned3BPDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.input2_nc = self.opt.input2_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            C (tensor) - - an alternative image in the input domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            C_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        # split AB image into A and B
        w, h = ABC.size
        w3 = int(w / 3)
        A = ABC.crop((0, 0, w3, h))
        B = ABC.crop((w3, 0, w3*2, h))
        C = ABC.crop((w3*2, 0, w, h))
        C = ImageOps.flip(C)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), convert=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), convert=False)
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input2_nc == 1), convert=False)

        srgb_img = A_transform(A)
        gt_SH = B_transform(B)
        L = C_transform(C)

        rgb_img = srgb_img**2.2
        gt_AL = torch.clamp(rgb_img / torch.clamp(gt_SH, min=1e-6), max=1.0, min=0.0)

        # gt_SH_gray = torch.mean(gt_SH, 0, keepdim=True)
        # mask = torch.ones_like(L)
        # mask[gt_SH_gray < 1e-4] = 0
        # mask = 1.0 - util.erosion(1.0-mask)

        # gt_BA, brightest_20, gt_BP, gt_BC\
        #      = util.calc_brightest(
        #          gt_SH_gray, mask,
        #          nr_tap=self.opt.bp_nr_tap, 
        #          nr_sigma=self.opt.bp_nr_sigma,
        #          spread_tap=self.opt.bp_tap, 
        #          spread_sigma=self.opt.bp_sigma
        #          )

        # if self.opt.shading_norm:
        #     gt_SH = gt_SH/brightest_20

        # srgb_img = normalize()(srgb_img)
        # gt_AL = normalize()(gt_AL)
        # gt_SH = normalize()(gt_SH)
        # mask = normalize(grayscale=True)(mask)
        # gt_BA = normalize(grayscale=True)(gt_BA)
        # gt_BP = normalize(grayscale=True)(gt_BP)
        # gt_BC = torch.Tensor(list(gt_BC))

        # srgb_img = torch.unsqueeze(srgb_img, 0) # [1, 3, 256, 256]
        # gt_AL = torch.unsqueeze(gt_AL, 0)
        # gt_SH = torch.unsqueeze(gt_SH, 0)
        # mask = torch.unsqueeze(mask, 0)
        # gt_BA = torch.unsqueeze(gt_BA, 0)
        # gt_BP = torch.unsqueeze(gt_BP, 0)

        res = self.make_bp_data(srgb_img, gt_SH, mask, self.opt, gt_AL=gt_AL)

        res['L'] = torch.unsqueeze(L, 0)
        res['A_paths'] = ABC_path

        # res(dict): A(srgb_img), gt_AL, gt_SH, gt_BA, gt_BP, gt_BC, L, mask
        # return {'A': res['srgb_img'], 'gt_AL': res['gt_AL'], 'gt_SH': res['gt_SH'], 'mask': mask, 'gt_BA': gt_BA, 'gt_BP': gt_BP, 'gt_BC':gt_BC, 'L': L, 'A_paths': ABC_path}
        return res

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
