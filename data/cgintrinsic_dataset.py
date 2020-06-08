import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pickle
import torch
import torch.nn as nn
import skimage
from skimage.restoration import denoise_tv_chambolle

# import skimage
# from skimage.morphology import square


# import os.path
# from data.base_dataset import BaseDataset, get_params, get_transform
# # from data.image_folder import make_dataset
# from PIL import Image
# import random


# def make_dataset(list_dir):
#     file_name = list_dir + "img_batch.p"
#     images_list = pickle.load( open( file_name, "rb" ) )

#     return images_list

# class CGIntrinsicsImageFolder(data.Dataset):

#     def __init__(self, root, list_dir, transform=None, 
#                  loader=None):
#         # load image list from hdf5
#         img_list = make_dataset(list_dir)
#         if len(img_list) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#         self.root = root
#         self.list_dir = list_dir
#         self.img_list = img_list
#         self.transform = transform
#         self.loader = loader
#         self.num_scale  = 4
#         self.sigma_n = 0.02
#         self.half_window = 1
#         self.height = 384
#         self.width = 512
#         self.original_h = 480
#         self.original_w = 640
#         self.rotation_range = 5.0
#         x = np.arange(-1, 2)
#         y = np.arange(-1, 2)
#         self.X, self.Y = np.meshgrid(x, y)
#         self.stat_dict = {}
#         f = open(self.root + "/CGIntrinsics/intrinsics_final/rgbe_image_stats.txt","r")
#         line = f.readline()
#         while line:
#             line = line.split()
#             self.stat_dict[line[0]] = float(line[2])
#             line = f.readline()

#     def DA(self, img, mode, random_pos, random_filp):

#         if random_filp > 0.5:
#             img = np.fliplr(img)

#         # img = rotate(img,random_angle, order = mode)
#         img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
#         img = resize(img, (self.height, self.width), order = mode)

#         return img

#     def construst_S_weights(self, normal):

#         center_feature = np.repeat( np.expand_dims(normal[4,:,:,:], axis =0), 9, axis = 0)
#         feature_diff = center_feature - normal

#         r_w = np.exp( - np.sum( feature_diff[:,:,:,0:3]**2  , 3) / (self.sigma_n**2))

#         return r_w

#     def construst_sub_matrix(self, C):
#         h = C.shape[0]
#         w = C.shape[1]

#         sub_C = np.zeros( (9 ,h-2,w-2, 3))
#         ct_idx = 0
#         for k in range(0, self.half_window*2+1):
#             for l in range(0,self.half_window*2+1):
#                 sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
#                 self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :] 
#                 ct_idx += 1

#         return sub_C

#     def load_CGIntrinsics(self, path):

#         img_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + path
#         srgb_img = np.float32(io.imread(img_path))/ 255.0
#         file_name = path.split('/')

#         R_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + file_name[0] + "/" + file_name[1][:-4] + "_albedo.png"
#         gt_R = np.float32(io.imread(R_path))/ 255.0

#         mask_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + file_name[0] + "/" + file_name[1][:-4] + "_mask.png"
#         mask = np.float32(io.imread(mask_path))/ 255.0
        
#         gt_R_gray = np.mean(gt_R, 2)
#         mask[gt_R_gray < 1e-6] = 0 
#         mask[np.mean(srgb_img,2) < 1e-6] = 0 

#         mask = skimage.morphology.binary_erosion(mask, square(11))
#         mask = np.expand_dims(mask, axis = 2)
#         mask = np.repeat(mask, 3, axis= 2)
#         gt_R[gt_R <1e-6] = 1e-6

#         # do normal DA
#         # random_angle = random.random() * self.rotation_range * 2.0 - self.rotation_range # random angle between -5 --- 5 degree
#         random_filp = random.random()
#         random_start_y = random.randint(0, 9) 
#         random_start_x = random.randint(0, 9) 

#         random_pos = [random_start_y, random_start_y + self.original_h - 10, random_start_x, random_start_x + self.original_w - 10]

#         srgb_img = self.DA(srgb_img, 1, random_pos, random_filp)
#         gt_R = self.DA(gt_R, 1,  random_pos, random_filp)
#         # cam_normal = self.DA(cam_normal, 0,  random_pos, random_filp)
#         mask = self.DA(mask, 0,  random_pos, random_filp)
#         rgb_img = srgb_img**2.2
#         gt_S = rgb_img / gt_R

#         search_name = path[:-4] + ".rgbe"
#         irridiance = self.stat_dict[search_name]

#         if irridiance < 0.25:
#             srgb_img = denoise_tv_chambolle(srgb_img, weight=0.05, multichannel=True)            
#             gt_S = denoise_tv_chambolle(gt_S, weight=0.1, multichannel=True)

#         mask[gt_S > 10] = 0
#         gt_S[gt_S > 20] = 20
#         mask[gt_S < 1e-4] = 0
#         gt_S[gt_S < 1e-4] = 1e-4

#         if np.sum(mask) < 10:
#             max_S = 1.0
#         else:
#             max_S = np.percentile(gt_S[mask > 0.5], 90)

#         gt_S = gt_S/max_S

#         gt_S = np.mean(gt_S, 2)
#         gt_S = np.expand_dims(gt_S, axis = 2)

#         gt_R = np.mean(gt_R,2)
#         gt_R = np.expand_dims(gt_R, axis = 2)

#         return srgb_img, gt_R, gt_S, mask, random_filp

#     def __getitem__(self, index):
#         targets_1 = {}
#         img_path = self.img_list[index]
#         # split_img_path = img_path.split('/')
#         full_path = self.root + img_path

#         srgb_img, gt_R, gt_S, mask, random_filp = self.load_CGIntrinsics(img_path)

#         targets_1['CGIntrinsics_ordinal_path'] = img_path
#         targets_1['random_filp'] = random_filp > 0.5

#         rgb_img = srgb_img**2.2
#         rgb_img[rgb_img < 1e-4] = 1e-4
#         chromaticity = rgb_to_chromaticity(rgb_img)
#         targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()

#         targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()
#         final_img = torch.from_numpy(np.transpose(srgb_img, (2, 0, 1))).contiguous().float()
#         targets_1['mask'] = torch.from_numpy( np.transpose(mask, (2 , 0 ,1))).contiguous().float()
#         targets_1['gt_R'] = torch.from_numpy(np.transpose(gt_R, (2 , 0 ,1))).contiguous().float()
#         targets_1['gt_S'] = torch.from_numpy(np.transpose(gt_S, (2 , 0 ,1))).contiguous().float()
#         targets_1['path'] = full_path

#         sparse_path_1s = self.root + "/CGIntrinsics/intrinsics_final/sparse_hdf5_S/384x512/R0.h5"

#         return final_img, targets_1, sparse_path_1s

#     def __len__(self):
#         return len(self.img_list)



# class CGIntrinsicsData(object):
#     def __init__(self, data_loader, root):
#         self.data_loader = data_loader
#         # self.fineSize = fineSize
#         # self.max_dataset_size = max_dataset_size
#         self.root = root
#         # st()
#         self.npixels = (256 * 256* 29)

#     def __iter__(self):
#         self.data_loader_iter = iter(self.data_loader)
#         self.iter = 0
#         return self

#     def sparse_loader(self, sparse_path, num_features):
#         # print("sparse_path  ", sparse_path)
#         # sys.exit()
#         hdf5_file_sparse = h5py.File(sparse_path,'r')
#         B_arr = []
#         data_whole = hdf5_file_sparse.get('/sparse/mn')
#         mn = np.array(data_whole)
#         mn = np.transpose(mn, (1,0))
#         m = int(mn[0][0])
#         n = int(mn[1][0])
#         # print(m, n)
#         data_whole = hdf5_file_sparse.get('/sparse/S')
#         S_coo = np.array(data_whole)
#         S_coo = np.transpose(S_coo, (1,0))
#         S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

#         # print(S_coo[:,0:2])
#         # print(torch.FloatTensor([3, 4]))
#         S_i = S_coo[0:2,:].long()
#         S_v = S_coo[2,:].float()
#         S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

#         for i in range(num_features+1):
#             data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
#             B_coo = np.array(data_whole)
#             B_coo = np.transpose(B_coo, (1,0))
#             B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
#             B_i = B_coo[0:2,:].long()
#             B_v = B_coo[2,:].float()

#             B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
#             B_arr.append(B_mat)


#         data_whole = hdf5_file_sparse.get('/sparse/N')
#         N = np.array(data_whole)
#         N = np.transpose(N, (1,0))
#         N = torch.from_numpy(N)

#         hdf5_file_sparse.close()
#         return S, B_arr, N 


#     def create_CGIntrinsics_pair(self, path, gt_albedo, random_filp):

#         super_pixel_path = self.root + "/CGIntrinsics/intrinsics_final/superpixels/" + path + ".mat"
#         super_pixel_mat = sio.loadmat(super_pixel_path)
#         super_pixel_mat = super_pixel_mat['data']
        
#         final_list = []

#         for i in range(len(super_pixel_mat)):
#             pos =super_pixel_mat[i][0]

#             if pos.shape[0] < 2:
#                 continue

#             rad_idx = random.randint(0, pos.shape[0]-1)            
#             final_list.append( (pos[rad_idx,0], pos[rad_idx,1]) )

#         eq_list = []
#         ineq_list = []

#         row = gt_albedo.shape[0]
#         col = gt_albedo.shape[1]

#         for i in range(0,len(final_list)-1):
#             for j in range(i+1, len(final_list)):
#                 y_1, x_1 = final_list[i]
#                 y_2, x_2 = final_list[j]

#                 y_1 = int(y_1*row)
#                 x_1 = int(x_1*col)
#                 y_2 = int(y_2*row)
#                 x_2 = int(x_2*col)

#                 # if image is flip
#                 if random_filp:
#                     x_1 = col - 1 - x_1
#                     x_2 = col - 1 - x_2

#                 if gt_albedo[y_1, x_1] < 2e-4 or gt_albedo[y_2, x_2] < 2e-4:
#                     continue

#                 ratio = gt_albedo[y_1, x_1]/gt_albedo[y_2, x_2]

#                 if ratio < 1.05 and ratio > 1./1.05:
#                     eq_list.append([y_1, x_1, y_2, x_2])
#                 elif ratio > 1.5:
#                     ineq_list.append([y_1, x_1, y_2, x_2])               
#                 elif ratio < 1./1.5:
#                     ineq_list.append([y_2, x_2, y_1, x_1])               

#         eq_mat = np.asarray(eq_list)
#         ineq_mat = np.asarray(ineq_list)

#         if eq_mat.shape[0] > 0:
#             eq_mat = torch.from_numpy(eq_mat).contiguous().float()
#         else:
#             eq_mat = torch.Tensor(1,1)


#         if ineq_mat.shape[0] > 0:
#             ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
#         else:
#             ineq_mat = torch.Tensor(1,1)


#         return eq_mat, ineq_mat


#     def __next__(self):
#         self.iter += 1
#         self.iter += 1
#         scale =4 

#         final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

#         target_1['eq_mat'] = []
#         target_1['ineq_mat'] = []
        
#         # This part will make training much slower, but it will improve performance
#         for i in range(len(target_1["CGIntrinsics_ordinal_path"])):
#             mat_path = target_1["CGIntrinsics_ordinal_path"][i]
#             gt_R = target_1['gt_R'][i,0,:,:].numpy()
#             random_filp = target_1['random_filp'][i]

#             eq_mat, ineq_mat = self.create_CGIntrinsics_pair(mat_path, gt_R, random_filp)
#             target_1['eq_mat'].append(eq_mat)
#             target_1['ineq_mat'].append(ineq_mat)


#         target_1['SS'] = []
#         target_1['SB_list'] = [] 
#         target_1['SN'] = []

#         SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[0], 2)

#         for i in range(len(sparse_path_1s)):
#             target_1['SS'].append(SS_1)
#             target_1['SB_list'].append(SB_list_1)
#             target_1['SN'].append(SN_1)

#         return {'img_1': final_img, 'target_1': target_1}



# class CGIntrinsics_DataLoader(BaseDataLoader):
#     def __init__(self,_root, _list_dir):
#         transform = None
#         dataset = CGIntrinsicsImageFolder(root=_root, \
#                 list_dir =_list_dir)

#         self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle= True, num_workers=int(2))
#         self.dataset = dataset
#         flip = False    
#         self.paired_data = CGIntrinsicsData(self.data_loader, _root)

#     def name(self):
#         return 'CGIntrinsics_DataLoader'

#     def load_data(self):
#         return self.paired_data

#     def __len__(self):
#         return len(self.dataset)


# def CreateDataLoaderCGIntrinsics(_root, _list_dir):
#     data_loader = None
#     from data.aligned_data_loader import CGIntrinsics_DataLoader
#     data_loader = CGIntrinsics_DataLoader(_root, _list_dir)
#     return data_loader


from typing import Union

import torch
import numpy as np


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

def make_dataset(list_dir, max_dataset_size=float("inf")):
    file_name = list_dir + "img_batch.p"
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
        self.img_paths = make_dataset(list_dir, opt.max_dataset_size)
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
        
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.erosion = nn.MaxPool2d(5, stride=1, padding=2)

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
        gt_R = Image.open(R_path).convert('RGB')

        mask_path = self.dataroot + "/intrinsics_final/images/"  + file_name[0] + "/" + file_name[1][:-4] + "_mask.png"
        mask = Image.open(mask_path).convert('RGB')

        irradiance = self.stat_dict[self.img_paths[index][:-4]+'.rgbe']
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, srgb_img.size)
        srgb_img_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        gt_R_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        mask_transform = get_transform(self.opt, transform_params, grayscale=True, convert=False)

        srgb_img = srgb_img_transform(srgb_img)
        gt_R = gt_R_transform(gt_R)
        mask = mask_transform(mask)

        gt_S = srgb_img**2.2 / torch.clamp(gt_R, min=1e-6)

        srgb_img_gray = torch.mean(srgb_img, 0, keepdim=True)
        gt_R_gray = torch.mean(gt_R, 0, keepdim=True)
        gt_S_gray = torch.mean(gt_S, 0, keepdim=True)

        gt_R[gt_R < 1e-6] = 1e-6
        mask[gt_R_gray < 1e-6] = 0 
        mask[srgb_img_gray < 1e-6] = 0 
        mask[gt_S_gray > 10] = 0
        gt_S[gt_S_gray.expand(gt_S.size()) > 20] = 20
        mask[gt_S_gray < 1e-4] = 0
        gt_S[gt_S_gray.expand(gt_S.size()) < 1e-4] = 1e-4

        mask = 1.0 - self.erosion(1.0-mask)

        brightest_point = percentile(gt_S[mask.expand(gt_S.size()) > 0.5], 90)
        # print('brightest_point:', brightest_point)

        brightest = torch.zeros_like(mask)
        brightest[gt_S_gray > brightest_point] = 1.0

        if self.opt.shading_norm:
            if torch.sum(mask) < 10:
                max_S = 1.0
            else:
                max_S = brightest_point
            gt_S = gt_S/max_S

        if irradiance < 0.25:
            srgb_img = srgb_img.cpu().numpy()
            gt_S = gt_S.cpu().numpy()
            srgb_img = denoise_tv_chambolle(srgb_img, weight=0.05, multichannel=True)            
            gt_S = denoise_tv_chambolle(gt_S, weight=0.1, multichannel=True)
            srgb_img = torch.from_numpy(srgb_img)
            gt_S = torch.from_numpy(gt_S)

        srgb_img = normalize()(srgb_img)
        gt_R = normalize()(gt_R)
        gt_S = normalize()(gt_S)
        mask = normalize(grayscale=True)(mask)
        brightest = normalize(grayscale=True)(brightest)

        srgb_img = torch.unsqueeze(srgb_img, 0) # [1, 3, 256, 256]
        gt_R = torch.unsqueeze(gt_R, 0)
        gt_S = torch.unsqueeze(gt_S, 0)
        mask = torch.unsqueeze(mask, 0)
        brightest = torch.unsqueeze(brightest, 0)
        
        # return {'A': srgb_img, 'B': gt_R, 'C': gt_S, 'D': mask, 'A_paths': img_path}
        return {'A': srgb_img, 'B': gt_R, 'C': gt_S, 'D': mask, 'E': brightest, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.img_paths)