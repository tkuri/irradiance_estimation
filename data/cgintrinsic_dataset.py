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

def make_dataset(list_dir, max_dataset_size=float("inf"), phase='train'):
    file_name = list_dir + "img_batch_{}.p".format(phase)
    print('load image list:', file_name)
    images_list = pickle.load( open( file_name, "rb" ) )

    return images_list[:min(max_dataset_size, len(images_list))]

class CGIntrinsicDataset(BaseDataset):
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
        self.dataroot = opt.dataroot # ../CGIntrinsics/CGIntrinsics
        list_dir = self.dataroot + '/intrinsics_final/train_list/'
        self.img_paths = make_dataset(list_dir, opt.max_dataset_size, opt.phase)
        if len(self.img_paths) == 0:
            raise(RuntimeError("Found 0 images in: " + list_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        # irradiance scale
        self.stat_dict = {}
        f = open(self.dataroot + "/intrinsics_final/rgbe_image_stats.txt","r")
        line = f.readline()
        while line:
            line = line.split()
            self.stat_dict[line[0]] = float(line[2])
            line = f.readline()

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
            
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
        img_path = self.dataroot + "/intrinsics_final/images/" + self.img_paths[index]
        srgb_img = Image.open(img_path).convert('RGB')
        file_name = self.img_paths[index].split('/')

        R_path = self.dataroot + "/intrinsics_final/images/"  + file_name[0] + "/" + file_name[1][:-4] + "_albedo.png"
        gt_AL = Image.open(R_path).convert('RGB')

        mask_path = self.dataroot + "/intrinsics_final/images/"  + file_name[0] + "/" + file_name[1][:-4] + "_mask.png"
        mask = Image.open(mask_path).convert('RGB')

        irradiance = self.stat_dict[self.img_paths[index][:-4]+'.rgbe']
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, srgb_img.size)
        srgb_img_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        gt_AL_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        mask_transform = get_transform(self.opt, transform_params, grayscale=True, convert=False)

        srgb_img = srgb_img_transform(srgb_img)
        gt_AL = gt_AL_transform(gt_AL)
        mask = mask_transform(mask)

        rgb_img = srgb_img**2.2
        gt_SH = rgb_img / torch.clamp(gt_AL, min=1e-6)

        srgb_img_gray = torch.mean(srgb_img, 0, keepdim=True)
        rgb_img_gray = torch.mean(rgb_img, 0, keepdim=True)
        gt_AL_gray = torch.mean(gt_AL, 0, keepdim=True)
        gt_SH_gray = torch.mean(gt_SH, 0, keepdim=True)

        gt_AL[gt_AL < 1e-6] = 1e-6
        # gt_SH[gt_SH_gray.expand(gt_SH.size()) > 20] = 20
        gt_SH[gt_SH_gray.expand(gt_SH.size()) > 1] = 1
        gt_SH[gt_SH_gray.expand(gt_SH.size()) < 1e-4] = 1e-4

        if not self.opt.no_mask:
            mask[srgb_img_gray < 1e-6] = 0 
            mask[gt_AL_gray < 1e-6] = 0 
            mask[gt_SH_gray < 1e-4] = 0
            mask[gt_SH_gray > 10] = 0
            mask = 1.0 - util.erosion(1.0-mask)

        mask_edge = mask.clone()

        if self.opt.edge_mask:
            edge_w = int(mask_edge.size(-1)*0.05)
            edge_h = int(mask_edge.size(-2)*0.05)
            mask_edge[:, :edge_h, :] = 0
            mask_edge[:, -edge_h:, :] = 0
            mask_edge[:, :, :edge_w] = 0
            mask_edge[:, :, -edge_w:] = 0

        gt_BA, brightest_20, gt_BP, gt_BC\
             = util.calc_brightest(
                 gt_SH_gray, mask_edge,
                 nr_tap=self.opt.bp_nr_tap, 
                 nr_sigma=self.opt.bp_nr_sigma,
                 spread_tap=self.opt.bp_tap, 
                 spread_sigma=self.opt.bp_sigma
                 )

        if self.opt.shading_norm:
            gt_SH = gt_SH/brightest_20

        if irradiance < 0.25:
            srgb_img = srgb_img.cpu().numpy()
            gt_SH = gt_SH.cpu().numpy()
            srgb_img = denoise_tv_chambolle(srgb_img, weight=0.05, multichannel=True)            
            gt_SH = denoise_tv_chambolle(gt_SH, weight=0.1, multichannel=True)
            srgb_img = torch.from_numpy(srgb_img)
            gt_SH = torch.from_numpy(gt_SH)

        srgb_img = normalize()(srgb_img)
        gt_AL = normalize()(gt_AL)
        gt_SH = normalize()(gt_SH)
        mask = normalize(grayscale=True)(mask)
        mask_edge = normalize(grayscale=True)(mask_edge)
        gt_BA = normalize(grayscale=True)(gt_BA)
        gt_BP = normalize(grayscale=True)(gt_BP)
        gt_BC = torch.Tensor(list(gt_BC))

        srgb_img = torch.unsqueeze(srgb_img, 0) # [1, 3, 256, 256]
        gt_AL = torch.unsqueeze(gt_AL, 0)
        gt_SH = torch.unsqueeze(gt_SH, 0)
        mask = torch.unsqueeze(mask, 0)
        mask_edge = torch.unsqueeze(mask_edge, 0)
        gt_BA = torch.unsqueeze(gt_BA, 0)
        gt_BP = torch.unsqueeze(gt_BP, 0)        
        # radiantest = torch.unsqueeze(radiantest, 0)
        
        # return {'A': srgb_img, 'B': gt_AL, 'C': gt_SH, 'D': mask, 'E': gt_BA, 'F': gt_BA, 'G': radiantest, 'A_paths': img_path}
        return {'A': srgb_img, 'gt_AL': gt_AL, 'gt_SH': gt_SH, 'mask': mask, 'mask_edge': mask_edge, 'gt_BA': gt_BA, 'gt_BP': gt_BP, 'gt_BC':gt_BC, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.img_paths)