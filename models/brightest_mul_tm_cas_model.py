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

class BrightestMulTmCasModel(BaseModel):
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
        parser.add_argument('--latent_Ls', action='store_true', help='Input Ls as latent.')
        parser.add_argument('--latent_Lt', action='store_true', help='Input Lt as latent.')
        parser.add_argument('--in_Ls', action='store_true', help='Input Ls as Input.')
        parser.add_argument('--in_Lt', action='store_true', help='Input Lt as Input.')
        parser.add_argument('--LTM', action='store_true', help='Use LTM.')
        parser.add_argument('--cat_In', action='store_true', help='Concat Input')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # self.loss_names = ['G_SH', 'G_BA', 'G_BP', 'G_BC']
        # self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'gt_BA', 'pr_BP', 'pr_BP2', 'gt_BP', 'pr_SH', 'gt_SH', 'mask']
        self.loss_names = ['G_SH', 'G_BA2', 'G_BC2']
        self.visual_names = ['input', 'pr_BA2', 'gt_BA', 'pr_SH', 'gt_SH']
        # self.visual_names = ['input', 'pr_BA2', 'gt_BA', 'pr_SH', 'gt_SH', 'L_itp']

        # self.model_names = ['G1', 'G2', 'G3']
        self.model_names = ['G1', 'G3']

        self.light_res = opt.light_res

        if opt.latent_Ls or opt.latent_Lt:
            netG1name = 'unet_256_latent_inL'
        else:
            netG1name = 'unet_256_latent'

        input_nc = opt.input_nc
        if opt.in_Ls:
            input_nc += 1
        if opt.in_Lt:
            input_nc += 1

        if opt.LTM:
            self.netG1 = networks.define_G(input_nc, self.light_res**2, opt.ngf, netG1name, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG1 = networks.define_G(input_nc, 1, opt.ngf, netG1name, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        g3_input_nc = opt.input_nc
        if opt.cat_In:
            g3_input_nc = g3_input_nc + 3
        self.netG3 = networks.define_G(g3_input_nc, 1, opt.ngf, 'resnet_9blocks_latent', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionS = torch.nn.MSELoss()
            self.criterionBA = torch.nn.MSELoss()
            # self.criterionBP = torch.nn.MSELoss()
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
        # self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BC = [torch.squeeze(input['gt_BC'][i],0).to(self.device) for i in range(25)] 

        if self.opt.in_Ls:
            self.Ls = torch.squeeze(input['Ls'],0).to(self.device) # [bn, 1, 256, 256]
        if self.opt.in_Lt:
            self.Lt = torch.squeeze(input['Lt'],0).to(self.device) # [bn, 1, 256, 256]

        self.Ls_stat = torch.squeeze(input['Ls_stat'],0).to(self.device) # [bn, 1, 256, 256]
        self.Lt_stat = torch.squeeze(input['Lt_stat'],0).to(self.device) # [bn, 1, 256, 256]

    def netG1_module(self):
        input_cat = self.input
        if self.opt.in_Ls:
            input_cat = torch.cat((input_cat, self.Ls), 1)
        if self.opt.in_Lt:
            input_cat = torch.cat((input_cat, self.Lt), 1)

        if self.opt.latent_Ls:
            dst, color = self.netG1(input_cat, self.Ls_stat.squeeze(-1)) # [25, 25, 256, 256]
        elif self.opt.latent_Lt:    
            dst, color = self.netG1(input_cat, self.Lt_stat.squeeze(-1)) # [25, 25, 256, 256]
        else:
            dst, color = self.netG1(input_cat) # [25, 25, 256, 256]

        return dst, color
        

    def ltm_module(self):
        ltm, color = netG1_module()
        # input_cat = self.input
        # if self.opt.in_Ls:
        #     input_cat = torch.cat((input_cat, self.Ls), 1)
        # if self.opt.in_Lt:
        #     input_cat = torch.cat((input_cat, self.Lt), 1)

        # if self.opt.latent_Ls:
        #     ltm, color = self.netG1(input_cat, self.Ls_stat.squeeze(-1)) # [25, 25, 256, 256]
        # elif self.opt.latent_Lt:    
        #     ltm, color = self.netG1(input_cat, self.Lt_stat.squeeze(-1)) # [25, 25, 256, 256]
        # else:
        #     ltm, color = self.netG1(input_cat) # [25, 25, 256, 256]

        ltm = ltm.view(-1, self.light_res**2, (ltm.size(-1)*ltm.size(-2)))  # [25, 25, 256x256]
        ltm = torch.transpose(ltm, 1, 2)  # [25, 256x256, 25]
        # ltm = torch.matmul(ltm, self.L) # L:[25, 25, 1] -> ltm[25, 256x256, 1]
        ltm = torch.matmul(ltm, self.Lt_stat) # L:[25, 25, 1] -> ltm[25, 256x256, 1]
        ltm = torch.transpose(ltm, 1, 2) # [25, 1, 256x256]
        ltm = (ltm - 0.5) / 0.5
        ltm = torch.clamp(ltm, min=-1.0, max=1.0)
        # pr_SH = buf.view(self.gt_SH.size()) # [25, 1, 256, 256]
        pr_SH = ltm.view(ltm.size(0), ltm.size(1), self.gt_SH.size(-2), self.gt_SH.size(-1)) # [25, 1, 256, 256]
        return pr_SH, color # pr_SH: -1~1
        # return pr_SH


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.LTM:
            self.pr_SH, color = self.ltm_module()
        else:
            self.pr_SH, color = self.netG1_module()

        # self.pr_SH = self.ltm_module()
        self.pr_SH = self.pr_SH.repeat(1, 3, 1, 1)
        self.pr_SH = self.pr_SH * 0.5 + 0.5
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.pr_SH = self.pr_SH * color
        self.pr_SH = self.pr_SH * 2.0 - 1.0

        # self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)

        if self.opt.cat_In:
            g3_input = torch.cat((self.pr_SH, self.input), 1)
        else:
            g3_input = self.pr_SH

        # self.pr_BC2, self.pr_BA2, self.pr_BP2 = self.netG3(g3_input)
        self.pr_BC2, self.pr_BA2 = self.netG3(g3_input)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        # gt_BC = self.gt_BC[:,:,:2]
        # condition = int(self.gt_BC[:, 0, 2].item())
        # bc_num = int(self.gt_BC[:, 0, 3].item())

        self.loss_G_SH = self.criterionS(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_S
        # self.loss_G_BA = self.criterionBA(self.pr_BA*mask, self.gt_BA*mask) * self.opt.lambda_BA
        # self.loss_G_BP = self.criterionBP(self.pr_BP*mask, self.gt_BP*mask) * self.opt.lambda_BP  
        self.loss_G_BA2 = self.criterionBA(self.pr_BA2*mask, self.gt_BA*mask) * self.opt.lambda_BA
        # self.loss_G_BP2 = self.criterionBP(self.pr_BP2*mask, self.gt_BP*mask) * self.opt.lambda_BP  

        # self.loss_G = self.loss_G_SH + self.loss_G_BA + self.loss_G_BP + self.loss_G_BA2 + self.loss_G_BP2
        # self.loss_G = self.loss_G_SH + self.loss_G_BA2 + self.loss_G_BP2
        self.loss_G = self.loss_G_SH + self.loss_G_BA2

        # print('gt_BC.shape 1', self.gt_BC.shape)
        # print('gt_BC.shape 2', gt_BC.shape)
        # gt_BC = gt_BC[:,0].squeeze(1)
        # print('gt_BC.shape 3', gt_BC.shape)

        # gt_BC = torch.cat([gt_BC[i][0].unsqueeze(0) for i in range(25)], dim=0)
        # loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC)

        # loss_G_BC2 = []
        for i in range(25):
            gt_BC = self.gt_BC[i][:, :2]
            bc_num = int(self.gt_BC[i][0, 3].item())
            pr_BC2 = self.pr_BC2[i]
            loss_G_BC2 = util.min_loss_BC_NoBatch(pr_BC2, gt_BC, bc_num, self.criterionBC)
            self.loss_G_BC2 = loss_G_BC2 * self.opt.lambda_BC / 25.0
            self.loss_G += self.loss_G_BC2

        # self.loss_G_BC2 = loss_G_BC2 * self.opt.lambda_BC
        # self.loss_G += self.loss_G_BC2

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
        # visual_ret['pr_BP_BC2'] = util.get_current_BC(self.pr_BC2, self.pr_BP2, self.opt)
        # visual_ret['pr_BP_BP'] = util.get_current_BP(self.pr_BP, self.opt)
        # visual_ret['pr_BP_BP2'] = util.get_current_BP(self.pr_BP2, self.opt)
        return visual_ret

    def eval_label(self):
        label = ['idx', 'condition', 'gt_BC_num']
        label += self.label_base()['BC'] + self.label_sh()['BC'] + self.label_pr(False, '2')['BC']
        label += self.label_base()['bcDist'] + self.label_sh()['bcDist'] + self.label_pr(False, '2')['bcDist']
        label += self.label_base()['baMSE'] + self.label_sh()['baMSE'] + self.label_pr(False, '2')['baMSE']
        label += self.label_sh()['shEval']
        return label

    def eval_brightest_pixel(self, idx=0):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()

        result = []        
        for i in range(25):
            res = [idx]
            res_base = self.eval_bp_base(self.mask, self.gt_BA[i].unsqueeze(0), None, self.gt_BC[i].unsqueeze(0), self.input[i].unsqueeze(0))
            res_sh = self.eval_bp_sh(self.mask, self.gt_BA[i].unsqueeze(0), None, self.gt_BC[i].unsqueeze(0), self.pr_SH[i].unsqueeze(0))
            res_sh.update(self.eval_sh(self.mask, self.gt_SH[i], self.pr_SH[i]))
            res_pr2 = self.eval_bp_pr(self.mask, self.gt_BA[i].unsqueeze(0), None, self.gt_BC[i].unsqueeze(0), self.pr_BA2[i].unsqueeze(0), None, self.pr_BC2[i].unsqueeze(0), '2')

            label = self.eval_label()
            for l in label:
                if l in res_base:
                    res.append(res_base[l])
                if l in res_sh:
                    res.append(res_sh[l])
                if l in res_pr2:
                    res.append(res_pr2[l])

            result.append(res)

        return result
