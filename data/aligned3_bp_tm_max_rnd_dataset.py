import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, make_bp_data
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import torch
import random
from torch.nn import functional as F

class Aligned3BpTmMaxRndDataset(BaseDataset):
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
        random.seed(100)        

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
        print('ABC_path:', ABC_path)
        ABC = Image.open(ABC_path).convert('RGB')
        # split AB image into A and B
        w, h = ABC.size
        h25 = int(h / 25)
        w3 = int(w / 3)
        srgb_img = []
        gt_SH = []
        Ls = []
        Lt = []

        tidx_list = list(range(25))
        random.shuffle(tidx_list)

        # tidx = 12
        for i in range(25):
            # tidx = random.randrange(25)
            tidx = tidx_list[i]
            # A.append(ABC.crop((0, h25*i, w3, h25*(i+1))))
            srgb_img.append(ABC.crop((0, h25*tidx, w3, h25*(tidx+1))))
            gt_SH.append(ABC.crop((w3, h25*i, w3*2, h25*(i+1))))

            Lsrc = ImageOps.flip(ABC.crop((w3*2, h25*tidx, w, h25*(tidx+1))))
            Lsrc = Lsrc.convert("L")
            _, vmax = Lsrc.getextrema()
            Lsrc = Lsrc.point(lambda x: 0 if x < vmax else 255) 
            Ls.append(Lsrc)

            Ltar = ImageOps.flip(ABC.crop((w3*2, h25*i, w, h25*(i+1))))
            Ltar = Ltar.convert("L")
            _, vmax = Ltar.getextrema()
            Ltar = Ltar.point(lambda x: 0 if x < vmax else 255) 
            Lt.append(Ltar)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, srgb_img[0].size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), convert=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), convert=False)
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input2_nc == 1), convert=False)

        for i in range(25):
            srgb_img[i] = A_transform(srgb_img[i])
            gt_SH[i] = B_transform(gt_SH[i])
            Ls[i] = C_transform(Ls[i])
            Lt[i] = C_transform(Lt[i])

        Ls_stat = []
        Lt_stat = []
        for i in range(25):
            Ls_stat_tmp = F.interpolate(Ls[i].unsqueeze(0), (self.opt.light_res, self.opt.light_res), mode='bilinear', align_corners=False)
            Ls_stat.append(Ls_stat_tmp.squeeze(0).view(self.opt.light_res**2, 1))
            Lt_stat_tmp = F.interpolate(Lt[i].unsqueeze(0), (self.opt.light_res, self.opt.light_res), mode='bilinear', align_corners=False)
            Lt_stat.append(Lt_stat_tmp.squeeze(0).view(self.opt.light_res**2, 1))

        mask = torch.ones_like(Ls[0])
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

        Ls_cat = torch.cat([torch.unsqueeze(Ls[i], 0) for i in range(25)], dim=0)
        Ls_stat_cat = torch.cat([torch.unsqueeze(Ls_stat[i], 0) for i in range(25)], dim=0)
        Lt_cat = torch.cat([torch.unsqueeze(Lt[i], 0) for i in range(25)], dim=0)
        Lt_stat_cat = torch.cat([torch.unsqueeze(Lt_stat[i], 0) for i in range(25)], dim=0)
        
        result['A'] = srgb_img_cat
        result['gt_SH'] = gt_SH_cat
        result['Ls'] = Ls_cat
        result['Lt'] = Lt_cat
        result['Ls_stat'] = Ls_stat_cat
        result['Lt_stat'] = Lt_stat_cat
        result['mask'] = torch.unsqueeze(mask, 0)

        result['gt_BA'] = gt_BA_cat
        # result['gt_BP'] = gt_BP_cat
        # result['gt_BC'] = gt_BC_cat
        result['gt_BC'] = []
        for i in range(25):
            result['gt_BC'].append(res[i]['gt_BC'])
        
        result['A_paths'] = ABC_path

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
