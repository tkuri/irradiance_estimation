import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import torch

class Aligned4Dataset(BaseDataset):
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
        self.dir_ABCD = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABCD_paths = sorted(make_dataset(self.dir_ABCD, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.input2_nc = self.opt.input2_nc
        self.input3_nc = self.opt.input3_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            C (tensor) - - an alternative image in the input domain
            D (tensor) - - an alternative image in the input domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            C_paths (str) - - image paths (same as A_paths)
            D_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABCD_path = self.ABCD_paths[index]
        ABCD = Image.open(ABCD_path).convert('RGB')
        # split AB image into A and B
        w, h = ABCD.size
        w4 = int(w / 4)
        A = ABCD.crop((0, 0, w4, h))
        B = ABCD.crop((w4, 0, w4*2, h))
        C = ABCD.crop((w4*2, 0, w4*3, h))
        D = ABCD.crop((w4*3, 0, w, h))
        C = ImageOps.flip(C) # Flip light image

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input2_nc == 1), convert=False)
        D_transform = get_transform(self.opt, transform_params, grayscale=(self.input3_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        D = D_transform(D)

        A = torch.unsqueeze(A, 0) # [1, 3, 256, 256]
        B = torch.unsqueeze(B, 0)
        C = torch.unsqueeze(C, 0)
        D = torch.unsqueeze(D, 0)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'A_paths': ABCD_path, 'B_paths': ABCD_path, 'C_paths': ABCD_path, 'D_paths': ABCD_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABCD_paths)
