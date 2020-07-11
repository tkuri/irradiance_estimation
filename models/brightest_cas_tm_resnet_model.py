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

class BrightestCasTmResnetModel(BaseModel):
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
            parser.add_argument('--lambda_BP', type=float, default=1.0, help='weight for Brightest pixel loss')
            parser.add_argument('--lambda_BC', type=float, default=1.0, help='weight for Brightest coordinate loss')
        parser.add_argument('--cat_In', action='store_true', help='Concat Input')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_SH', 'G_BA', 'G_BP', 'G_BC']
        self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'gt_BA', 'pr_BP', 'pr_BP2', 'gt_BP', 'pr_SH', 'gt_SH', 'mask']

        # self.model_names = ['G1', 'G2', 'G3']
        self.model_names = ['G1', 'G3']

        self.light_res = opt.light_res
        # self.netG1 = networks.define_G(opt.input_nc, 3, opt.ngf, 'unet_256_multi', opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, self.light_res**2, opt.ngf, 'unet_256_latent', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        g3_input_nc = opt.input_nc
        if opt.cat_In:
            g3_input_nc = g3_input_nc + 3
        self.netG3 = networks.define_G(g3_input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionS = torch.nn.MSELoss()
            self.criterionBA = torch.nn.MSELoss()
            self.criterionBP = torch.nn.MSELoss()
            self.criterionBC = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            # self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_G3)

    def set_input(self, input):
        self.input = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.image_paths = input['A_paths']
        self.gt_SH = torch.squeeze(input['gt_SH'],0).to(self.device) # [bn, 3, 256, 256]
        self.mask = torch.squeeze(input['mask'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BA = torch.squeeze(input['gt_BA'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BC = [input['gt_BC'][i].to(self.device) for i in range(25)] 
        self.L = torch.squeeze(input['L'],0).to(self.device) # [bn, 1, 256, 256]
        self.L = F.interpolate(self.L, (self.light_res, self.light_res), mode='bilinear', align_corners=False) # [bn, 1, 5, 5]
        self.L = self.L.view(-1, self.light_res**2, 1) # [bn, 25, 1]
        # self.L_itp = torch.clamp((F.interpolate(self.L_itp, (self.L.size(-2), self.L.size(-1)), mode='nearest')-0.5)/0.5, min=-1.0, max=1.0)  # [bn, 256, 256, 1]

    def ltm_module(self):
        ltm, color = self.netG1(self.input) # [25, 25, 256, 256]
        ltm = ltm.view(-1, self.light_res**2, (ltm.size(-1)*ltm.size(-2)))  # [25, 3*16, 256x256]
        ltm = torch.transpose(ltm, 1, 2)  # [25, 256x256, 3*16]
        ltm = torch.matmul(ltm, self.L)
        ltm = torch.transpose(ltm, 1, 2) # [25, 1, 256x256]
        ltm = (ltm - 0.5) / 0.5
        ltm = torch.clamp(ltm, min=-1.0, max=1.0)
        # pr_SH = buf.view(self.gt_SH.size()) # [25, 1, 256, 256]
        pr_SH = ltm.view(ltm.size(0), ltm.size(1), self.gt_SH.size(-2), self.gt_SH.size(-1)) # [25, 1, 256, 256]
        return pr_SH, color


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        pr_SH, color = self.ltm_module()
        pr_SH = pr_SH.repeat(1, 3, 1, 1)
        pr_SH = pr_SH * 0.5 + 0.5
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.pr_SH = pr_SH * color
        self.pr_SH = self.pr_SH * 2.0 - 1.0
        # self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)

        if self.opt.cat_In:
            g3_input = torch.cat((self.pr_SH, self.input), 1)
        else:
            g3_input = self.pr_SH

        self.pr_BC2, self.pr_BA2, self.pr_BP2 = self.netG3(g3_input)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        # gt_BC = self.gt_BC[:,:,:2]
        gt_BC = [self.gt_BC[i][:,:2] for i in range(25)]
        # condition = int(self.gt_BC[:, 0, 2].item())
        # bc_num = int(self.gt_BC[:, 0, 3].item())

        self.loss_G_SH = self.criterionS(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_S
        # self.loss_G_BA = self.criterionBA(self.pr_BA*mask, self.gt_BA*mask) * self.opt.lambda_BA
        # self.loss_G_BP = self.criterionBP(self.pr_BP*mask, self.gt_BP*mask) * self.opt.lambda_BP  
        self.loss_G_BA2 = self.criterionBA(self.pr_BA2*mask, self.gt_BA*mask) * self.opt.lambda_BA
        self.loss_G_BP2 = self.criterionBP(self.pr_BP2*mask, self.gt_BP*mask) * self.opt.lambda_BP  

        # self.loss_G = self.loss_G_SH + self.loss_G_BA + self.loss_G_BP + self.loss_G_BA2 + self.loss_G_BP2
        self.loss_G = self.loss_G_SH + self.loss_G_BA2 + self.loss_G_BP2

        # print('gt_BC.shape 1', self.gt_BC.shape)
        # print('gt_BC.shape 2', gt_BC.shape)
        gt_BC = gt_BC[:,0].squeeze(1)
        # print('gt_BC.shape 3', gt_BC.shape)

        loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC[0])
        self.loss_G_BC2 = loss_G_BC2 * self.opt.lambda_BC
        self.loss_G += self.loss_G_BC2

        # print('condition:', condition)
        # if condition==1:
        #     # self.loss_G_BC = self.criterionBC(self.pr_BC, gt_BC.squeeze(1)) * self.opt.lambda_BC
        #     self.loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC.squeeze(1)) * self.opt.lambda_BC
        #     # self.loss_G += self.loss_G_BC + self.loss_G_BC2
        #     self.loss_G += self.loss_G_BC2
        # # else:
        # elif condition==2:
        #     # loss_G_BC = util.min_loss_BC(self.pr_BC, gt_BC, bc_num, self.criterionBC)
        #     # loss_G_BC2 = util.min_loss_BC(self.pr_BC2, gt_BC, bc_num, self.criterionBC)
        #     loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC[:,0].squeeze(1))

        #     # self.loss_G_BC = loss_G_BC * self.opt.lambda_BC
        #     self.loss_G_BC2 = loss_G_BC2 * self.opt.lambda_BC
        #     # self.loss_G += self.loss_G_BC + self.loss_G_BC2
        #     self.loss_G += self.loss_G_BC2
        # else:
        #     print('Pass loss_G_BC because condition is {}'.format(condition))

        self.loss_G.backward()

    def optimize_parameters(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        # self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.optimizer_G3.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G3.step()             # udpate G's weights
        self.optimizer_G1.step()             # udpate G's weights
        # self.optimizer_G2.step()             # udpate G's weights

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        # visual_ret['pr_BP_BC'] = util.get_current_BC(self.pr_BC, self.pr_BP, self.opt)
        visual_ret['pr_BP_BC2'] = util.get_current_BC(self.pr_BC2, self.pr_BP2, self.opt)
        # visual_ret['pr_BP_BP'] = util.get_current_BP(self.pr_BP, self.opt)
        visual_ret['pr_BP_BP2'] = util.get_current_BP(self.pr_BP2, self.opt)
        return visual_ret

    def eval_label(self):
        # label = ['idx', 'condition', 'bc_gt', 'bc_ra', 'bc_sh', 'bc_ba', 'bc_bp', 'bc_bc', 
        # 'bc_ba2', 'bc_bp2', 'bc_bc2', 
        # 'dist_ra', 'dist_sh', 'dist_ba', 'dist_bp', 'dist_bc',
        # 'dist_ba2', 'dist_bp2', 'dist_bc2', 'dist_05',
        # 'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba', 'ba_mse_ba2','ba_mse_0', 'ba_mse_h', 'ba_mse_1',
        # 'bp_mse_ra', 'bp_mse_sh', 'bp_mse_ba', 'bp_mse_bp', 'bp_mse_bp_direct', 
        # 'bp_mse_ba2', 'bp_mse_bp2', 'bp_mse_bp2_direct', 'bp_mse_0', 'bp_mse_h', 'bp_mse_1']
        label = ['idx', 'condition', 'bc_gt', 'bc_ra', 'bc_sh', 
        'bc_ba2', 'bc_bp2', 'bc_bc2', 
        'dist_ra', 'dist_sh',
        'dist_ba2', 'dist_bp2', 'dist_bc2', 'dist_05',
        'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba2','ba_mse_0', 'ba_mse_h', 'ba_mse_1',
        'bp_mse_ra', 'bp_mse_sh', 
        'bp_mse_ba2', 'bp_mse_bp2', 'bp_mse_bp2_direct', 'bp_mse_0', 'bp_mse_h', 'bp_mse_1']

        return label

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        
        res_base = self.eval_bp_base()
        res_cas = self.eval_bp_cas()

        result = []
        label = self.eval_label()
        for l in label:
            if l in res_base:
                result.append(res_base[l])
            if l in res_cas:
                result.append(res_cas[l])
        return result
