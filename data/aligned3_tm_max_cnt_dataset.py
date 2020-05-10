import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import torch

class Aligned3TmMaxCntDataset(BaseDataset):
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
        h25 = int(h / 25)
        w3 = int(w / 3)
        A = []
        B = []
        C = []

        tidx = 12
        
        for i in range(25):
            # A.append(ABC.crop((0, h25*i, w3, h25*(i+1))))
            A.append(ABC.crop((0, h25*tidx, w3, h25*(tidx+1))))
            B.append(ABC.crop((w3, h25*i, w3*2, h25*(i+1))))
            Ctmp = ImageOps.flip(ABC.crop((w3*2, h25*i, w, h25*(i+1))))
            Ctmp = Ctmp.convert("L")
            _, vmax = Ctmp.getextrema()
            Ctmp = Ctmp.point(lambda x: 0 if x < vmax else 255) 
            C.append(Ctmp)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A[0].size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input2_nc == 1), convert=False)

        for i in range(25):
            A[i] = A_transform(A[i])
            B[i] = B_transform(B[i])
            C[i] = C_transform(C[i])
            
        Acat = torch.unsqueeze(A[0], 0)
        Bcat = torch.unsqueeze(B[0], 0)
        Ccat = torch.unsqueeze(C[0], 0)
        for i in range(1,25):
            Acat = torch.cat([Acat, torch.unsqueeze(A[i], 0)], dim=0) # [25, 3, 256, 256]
            Bcat = torch.cat([Bcat, torch.unsqueeze(B[i], 0)], dim=0)
            Ccat = torch.cat([Ccat, torch.unsqueeze(C[i], 0)], dim=0)
        
        # print('Acat size:', Acat.size())
        # print('A_trans:', A.max(), A.min())
        # print('B_trans:', B.max(), B.min())
        # print('C_trans:', C.max(), C.min())

        return {'A': Acat, 'B': Bcat, 'C': Ccat, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
