import torch
from collections import OrderedDict
from .base_model import BaseModel
from . import networks
from typing import Union
from util import util
import numpy as np
from . import saw_utils
from skimage.transform import resize
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
import cv2
import json

class BrightestModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_SH', type=float, default=1.0, help='weight for Shading loss')
            parser.add_argument('--lambda_AL', type=float, default=1.0, help='weight for Reflection loss')
            parser.add_argument('--lambda_BA', type=float, default=1.0, help='weight for Brightest area loss')
            parser.add_argument('--lambda_BP', type=float, default=1.0, help='weight for Brightest pixel loss')
            parser.add_argument('--lambda_BC', type=float, default=1.0, help='weight for Brightest coordinate loss')
            parser.add_argument('--reg', action='store_true', help='regularization')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_AL', 'G_SH', 'G_BA', 'G_BP', 'G_BC']

        self.visual_names = ['input', 'pr_BA', 'gt_BA', 'pr_BP', 'gt_BP', 'pr_AL', 'gt_AL', 'pr_SH', 'gt_SH', 'mask']

        self.model_names = ['G1', 'G2']

        self.netG1 = networks.define_G(opt.input_nc, 3, opt.ngf, 'unet_256_multi', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionAL = torch.nn.MSELoss()
            self.criterionSH = torch.nn.MSELoss()
            self.criterionBA = torch.nn.MSELoss()
            self.criterionBP = torch.nn.MSELoss()
            self.criterionBC = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.image_paths = input['A_paths']
        self.gt_AL = torch.squeeze(input['gt_AL'],0).to(self.device) # [bn, 3, 256, 256]
        self.gt_SH = torch.squeeze(input['gt_SH'],0).to(self.device) # [bn, 3, 256, 256]
        self.mask = torch.squeeze(input['mask'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BA = torch.squeeze(input['gt_BA'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BC = input['gt_BC'].to(self.device) 
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pr_SH, pr_AL, color = self.netG1(self.input)  # G(A)
        self.pr_AL = pr_AL
        pr_SH = pr_SH.repeat(1, 3, 1, 1)
        pr_SH = pr_SH * 0.5 + 0.5
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.pr_SH = pr_SH * color
        self.pr_SH = self.pr_SH * 2.0 - 1.0
        self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        gt_BC = self.gt_BC[:,:,:2]
        condition = int(self.gt_BC[:, 0, 2].item())
        bc_num = int(self.gt_BC[:, 0, 3].item())
        self.loss_G_AL = self.criterionAL(self.pr_AL*mask, self.gt_AL*mask) * self.opt.lambda_AL
        self.loss_G_SH = self.criterionSH(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_SH
        self.loss_G_BA = self.criterionBA(self.pr_BA*mask, self.gt_BA*mask) * self.opt.lambda_BA
        self.loss_G_BP = self.criterionBP(self.pr_BP*mask, self.gt_BP*mask) * self.opt.lambda_BP  

        self.loss_G = self.loss_G_AL + self.loss_G_SH + self.loss_G_BA + self.loss_G_BP

        if condition==1:
            self.loss_G_BC = self.criterionBC(self.pr_BC, gt_BC.squeeze(1)) * self.opt.lambda_BC
            self.loss_G += self.loss_G_BC
        elif condition==2:
            loss_G_BC = self.criterionBC(self.pr_BC, gt_BC[:, 0])
            for i in range(1, bc_num):
                loss_G_BC_cmp = self.criterionBC(self.pr_BC, gt_BC[:, i].squeeze(1))
                loss_G_BC = torch.min(loss_G_BC, loss_G_BC_cmp)
            self.loss_G_BC = loss_G_BC * self.opt.lambda_BC
            self.loss_G += self.loss_G_BC
        else:
            print('Pass loss_G_BC because condition is {}'.format(condition))

        self.loss_G.backward()

    def optimize_parameters(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        visual_ret['pr_BP_BC'] = util.get_current_BC(self.pr_BC, self.pr_BP, self.opt)
        visual_ret['pr_BP_BP'] = util.get_current_BP(self.pr_BP, self.opt)
        return visual_ret


    def eval_label(self):
        label = ['idx', 'condition']
        label += self.label_base()['BC'] + self.label_sh()['BC'] + self.label_pr()['BC']
        label += self.label_base()['dict_BC'] + self.label_sh()['dict_BC'] + self.label_pr()['dict_BC']
        label += self.label_base()['mse_BA'] + self.label_sh()['mse_BA'] + self.label_pr()['mse_BA']
        label += self.label_base()['mse_BP'] + self.label_sh()['mse_BP'] + self.label_pr()['mse_BP']

        # label = ['idx', 'condition', 'gt_BC', 'base_BC_RA', 'pr_BC_BA', 'bc_ba', 'bc_bp', 'bc_bc', 
        # 'dist_ra', 'dist_sh', 'dist_ba', 'dist_bp', 'dist_bc', 'dist_05',
        # 'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba', 'ba_mse_0',
        # 'bp_mse_ra', 'bp_mse_sh', 'bp_mse_ba', 'bp_mse_bp', 'bp_mse_bp_direct', 'bp_mse_0']

        return label

    def eval_brightest_pixel(self, idx=0):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        
        res_base = self.eval_bp_base(self.mask, self.gt_BA, self.gt_BP, self.gt_BC, self.input)
        res_sh = self.eval_bp_sh(self.mask, self.gt_BA, self.gt_BP, self.gt_BC, self.pr_SH)
        res_pr = self.eval_bp_pr(self.mask, self.gt_BA, self.gt_BP, self.gt_BC, self.pr_BA, self.pr_BP, self.pr_BC, '')

        result = [idx]

        label = self.eval_label()
        for l in label:
            if l in res_base:
                result.append(res_base[l])
            if l in res_sh:
                result.append(res_sh[l])
            if l in res_pr:
                result.append(res_pr[l])
        return result