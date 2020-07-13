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
from torch.nn import functional as F

class BrightestMulInLModel(BaseModel):
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
            parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for Shading loss')
            parser.add_argument('--lambda_BA', type=float, default=1.0, help='weight for Brightest area loss')
            # parser.add_argument('--lambda_BP', type=float, default=1.0, help='weight for Brightest pixel loss')
            parser.add_argument('--lambda_BC', type=float, default=1.0, help='weight for Brightest coordinate loss')
        parser.add_argument('--cat_In', action='store_true', help='Concat Input')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_SH', 'G_BA', 'G_BC']
        # self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'gt_BA', 'pr_BP', 'pr_BP2', 'gt_BP', 'pr_SH', 'gt_SH', 'mask']
        self.visual_names = ['input', 'pr_BA', 'gt_BA', 'pr_SH', 'gt_SH', 'mask', 'L_itp']

        # self.model_names = ['G1', 'G2', 'G3']
        self.model_names = ['G1', 'G2']

        self.light_res = opt.light_res
        self.netG1 = networks.define_G(opt.input_nc + opt.input2_nc, 1, opt.ngf, 'unet_256_latent', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG2 = networks.define_G(opt.input_nc + opt.input2_nc, 1, opt.ngf, 'resnet_9blocks_latent', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionS = torch.nn.MSELoss()
            self.criterionBA = torch.nn.MSELoss()
            # self.criterionBP = torch.nn.MSELoss()
            self.criterionBC = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)

    def set_input(self, input):
        self.input = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.image_paths = input['A_paths']
        self.gt_SH = torch.squeeze(input['gt_SH'],0).to(self.device) # [bn, 3, 256, 256]
        self.mask = torch.squeeze(input['mask'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BA = torch.squeeze(input['gt_BA'],0).to(self.device) # [bn, 1, 256, 256]
        # self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BC = [torch.squeeze(input['gt_BC'][i],0).to(self.device) for i in range(25)] 
        self.L = torch.squeeze(input['L'],0).to(self.device) # [bn, 1, 256, 256]
        self.L_itp = torch.clamp((F.interpolate(self.L[0].unsqueeze(0), (self.L.size(-2), self.L.size(-1)), mode='nearest')-0.5)/0.5, min=-1.0, max=1.0)  # [bn, 256, 256, 1]
        # self.L = F.interpolate(self.L, (self.light_res, self.light_res), mode='bilinear', align_corners=False) # [bn, 1, 5, 5]
        # self.L = self.L.view(-1, self.light_res**2, 1) # [bn, 25, 1]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        L = self.L * 2.0 - 1.0
        in_cat = torch.cat([self.input, L], 1) 
        self.pr_SH, color = self.netG1(in_cat) # [bn, 1, 256, 256]
        # self.pr_SH, color = self.netG1(self.input) # [bn, 1, 256, 256]
        self.pr_SH = self.pr_SH.repeat(1, 3, 1, 1)
        self.pr_SH = self.pr_SH * 0.5 + 0.5
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.pr_SH = self.pr_SH * color
        self.pr_SH = self.pr_SH * 2.0 - 1.0

        # self.pr_BC, self.pr_BA = self.netG2(self.input)
        self.pr_BC, self.pr_BA = self.netG2(in_cat)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        # condition = int(self.gt_BC[:, 0, 2].item())

        self.loss_G_SH = self.criterionS(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_S
        self.loss_G_BA = self.criterionBA(self.pr_BA*mask, self.gt_BA*mask) * self.opt.lambda_BA
        # self.loss_G_BP = self.criterionBP(self.pr_BP*mask, self.gt_BP*mask) * self.opt.lambda_BP  

        # self.loss_G = self.loss_G_SH + self.loss_G_BA + self.loss_G_BP + self.loss_G_BA2 + self.loss_G_BP2
        self.loss_G = self.loss_G_SH + self.loss_G_BA

        for i in range(25):
            gt_BC = self.gt_BC[i][:, :2]
            bc_num = int(self.gt_BC[i][0, 3].item())
            pr_BC = self.pr_BC[i]
            loss_G_BC = util.min_loss_BC_NoBatch(pr_BC, gt_BC, bc_num, self.criterionBC)
            self.loss_G_BC = loss_G_BC * self.opt.lambda_BC / 25.0
            self.loss_G += self.loss_G_BC

        self.loss_G.backward()

    def optimize_parameters(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        # self.optimizer_G3.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # self.optimizer_G3.step()             # udpate G's weights
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        visual_ret['pr_BP_BC'] = util.get_current_BC(self.pr_BC, self.pr_BP, self.opt)
        return visual_ret

    def eval_label(self):
        label += self.label_base()['BC'] + self.label_sh()['BC'] + self.label_pr(False)['BC']
        label += self.label_base()['dict_BC'] + self.label_sh()['dict_BC'] + self.label_pr(False)['dict_BC']
        label += self.label_base()['mse_BA'] + self.label_sh()['mse_BA'] + self.label_pr(False)['mse_BA']
        label += self.label_base()['mse_BP'] + self.label_sh()['mse_BP'] + self.label_pr(False)['mse_BP']
        return label

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        
        result = []        
        for i in range(25):
            res = [idx]
            res_base = self.eval_bp_base(self.mask, self.ge_BA[i], self.gt_BP[i], self.gt_BC[i], self.input[i])
            res_sh = self.eval_bp_sh(self.mask, self.gt_BA[i], self.gt_BP[i], self.gt_BC[i], self.gt_SH[i])
            res_pr = self.eval_bp_pr(self.mask, self.gt_BA[i], self.gt_BP[i], self.gt_BC[i], self.pr_BA[i], None, '')

            res = []
            label = self.eval_label()
            for l in label:
                if l in res_base:
                    res.append(res_base[l])
                if l in res_sh:
                    res.append(res_sh[l])
                if l in res_pr:
                    res.append(res_pr[l])

            result.append(res)

        return result
