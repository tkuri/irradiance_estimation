from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import cv2
import os

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
        A_img = Image.open(A_path).convert('RGB')
        gtlx_np = np.loadtxt(gtlx_path, delimiter=',')
        gtlx_pil = Image.fromarray(gtlx_np)
        A = self.transform(A_img)
        gtlx = self.transform_lux(gtlx_pil)
        gtlx = gtlx / gtlx.max()
        A = A.unsqueeze(0)
        return {'A': A, 'gtlx':gtlx, 'A_paths': A_path, 'gtlx_paths': gtlx_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
