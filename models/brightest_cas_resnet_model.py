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

class BrightestCasResnetModel(BaseModel):
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
            parser.add_argument('--lambda_AL', type=float, default=1.0, help='weight for Reflection loss')
            parser.add_argument('--lambda_BA', type=float, default=1.0, help='weight for Brightest area loss')
            parser.add_argument('--lambda_BP', type=float, default=1.0, help='weight for Brightest pixel loss')
            parser.add_argument('--lambda_BC', type=float, default=1.0, help='weight for Brightest coordinate loss')
        parser.add_argument('--cat_AL', action='store_true', help='Concat AL')
        parser.add_argument('--cat_In_AL', action='store_true', help='Concat Input and AL')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_AL', 'G_SH', 'G_BA', 'G_BP', 'G_BC']
        if not opt.no_gt:
            self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'gt_BA', 'pr_BP', 'pr_BP2', 'gt_BP', 'pr_AL', 'gt_AL', 'pr_SH', 'gt_SH', 'mask']
        else:
            self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'pr_BP', 'pr_BP2', 'pr_AL', 'pr_SH']

        self.model_names = ['G1', 'G2', 'G3']

        self.netG1 = networks.define_G(opt.input_nc, 3, opt.ngf, 'unet_256_multi', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        g3_input_nc = opt.input_nc
        if opt.cat_AL:
            g3_input_nc = g3_input_nc + 3
        if opt.cat_In_AL:
            g3_input_nc = g3_input_nc + 6
        self.netG3 = networks.define_G(g3_input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionR = torch.nn.MSELoss()
            self.criterionS = torch.nn.MSELoss()
            self.criterionBA = torch.nn.MSELoss()
            self.criterionBP = torch.nn.MSELoss()
            self.criterionBC = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_G3)

    def set_input(self, input):
        self.input = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.image_paths = input['A_paths']
        if not self.opt.no_gt:
            self.gt_AL = torch.squeeze(input['gt_AL'],0).to(self.device) # [bn, 3, 256, 256]
            self.gt_SH = torch.squeeze(input['gt_SH'],0).to(self.device) # [bn, 3, 256, 256]
            self.mask = torch.squeeze(input['mask'],0).to(self.device) # [bn, 1, 256, 256]
            self.gt_BA = torch.squeeze(input['gt_BA'],0).to(self.device) # [bn, 1, 256, 256]
            self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
            self.gt_BC = input['gt_BC'].to(self.device) 
    
    def percentile(self, t: torch.tensor, q: float) -> Union[int, float]:
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def calc_shading(self, img, albedo, mask):
        img = torch.clamp(img * 0.5 + 0.5, min=0.0, max=1.0) # 0~1
        albedo = torch.clamp(albedo * 0.5 + 0.5, min=1e-6, max=1.0) # 0~1
        shading = img**2.2/albedo
        if self.opt.shading_norm:
            if torch.sum(mask) < 10:
                max_S = 1.0
            else:
                max_S = self.percentile(shading[self.mask.expand(shading.size()) > 0.5], 90)

            shading = shading/max_S

        shading = (shading - 0.5) / 0.5
        return torch.clamp(shading, min=-1.0, max=1.0) # -1~1

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pr_SH, pr_AL, color = self.netG1(self.input)  # G(A)
        self.pr_AL = pr_AL
        pr_SH = pr_SH.repeat(1, 3, 1, 1)
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.pr_SH = pr_SH * color
        self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)

        if self.opt.cat_AL:
            g3_input = torch.cat((self.pr_SH, self.pr_AL), 1)
        elif self.opt.cat_In_AL:
            g3_input = torch.cat((self.pr_SH, self.pr_AL), 1)
            g3_input = torch.cat((g3_input, self.input), 1)
        else:
            g3_input = self.pr_SH

        self.pr_BC2, self.pr_BA2, self.pr_BP2 = self.netG3(g3_input)

    def min_loss_BC(pr_BC, gt_BC):
        loss_G_BC = self.criterionBC(pr_BC, gt_BC[:, 0])
        for i in range(1, bc_num):
            loss_G_BC_cmp = self.criterionBC(pr_BC, gt_BC[:, i].squeeze(1))
            loss_G_BC = torch.min(loss_G_BC, loss_G_BC_cmp)
        return loss_G_BC

        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        gt_BC = self.gt_BC[:,:,:2]
        condition = int(self.gt_BC[:, 0, 2].item())
        bc_num = int(self.gt_BC[:, 0, 3].item())

        self.loss_G_AL = self.criterionR(self.pr_AL*mask, self.gt_AL*mask) * self.opt.lambda_AL
        self.loss_G_SH = self.criterionS(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_S
        self.loss_G_BA = self.criterionBA(self.pr_BA*mask, self.gt_BA*mask) * self.opt.lambda_BA
        self.loss_G_BP = self.criterionBP(self.pr_BP*mask, self.gt_BP*mask) * self.opt.lambda_BP  
        self.loss_G_BA2 = self.criterionBA(self.pr_BA2*mask, self.gt_BA*mask) * self.opt.lambda_BA
        self.loss_G_BP2 = self.criterionBP(self.pr_BP2*mask, self.gt_BP*mask) * self.opt.lambda_BP  

        self.loss_G = self.loss_G_AL + self.loss_G_SH + self.loss_G_BA + self.loss_G_BP + self.loss_G_BA2 + self.loss_G_BP2
        if condition==1:
            self.loss_G_BC = self.criterionBC(self.pr_BC, gt_BC.squeeze(1)) * self.opt.lambda_BC
            self.loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC.squeeze(1)) * self.opt.lambda_BC
            self.loss_G += self.loss_G_BC + self.loss_G_BC2
        # else:
        elif condition==2:
            loss_G_BC = self.min_loss_BC(self.pr_BC, gt_BC)
            loss_G_BC2 = self.min_loss_BC(self.pr_BC2, gt_BC)
            # loss_G_BC = self.criterionBC(self.pr_BC, gt_BC[:, 0])
            # for i in range(1, bc_num):
            #     loss_G_BC_cmp = self.criterionBC(self.pr_BC, gt_BC[:, i].squeeze(1))
            #     loss_G_BC = torch.min(loss_G_BC, loss_G_BC_cmp)

            # loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC[:, 0])
            # for i in range(1, bc_num):
            #     loss_G_BC2_cmp = self.criterionBC(self.pr_BC2, gt_BC[:, i].squeeze(1))
            #     loss_G_BC2 = torch.min(loss_G_BC, loss_G_BC2_cmp)

            self.loss_G_BC = loss_G_BC * self.opt.lambda_BC
            self.loss_G_BC2 = loss_G_BC2 * self.opt.lambda_BC
            self.loss_G += self.loss_G_BC + self.loss_G_BC2
        else:
            print('Pass loss_G_BC because condition is {}'.format(condition))

        self.loss_G.backward()

    def optimize_parameters(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.optimizer_G3.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G3.step()             # udpate G's weights
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights

    def get_current_BC(self, pr_BC):
        pr_BP_BC = util.disp_brightest_coord(pr_BC, self.pr_BP, self.opt.bp_tap, self.opt.bp_sigma)
        pr_BP_BC = (pr_BP_BC - 0.5) / 0.5
        return pr_BP_BC

    def get_current_BP(self, pr_BP):
        pr_BP_norm = torch.squeeze(pr_BP, 0)*0.5+0.5
        mask_one = torch.ones_like(pr_BP_norm)
        _, _, pr_BP_BP, _ = util.calc_brightest(pr_BP_norm, mask_one, self.opt.bp_nr_tap, self.opt.bp_nr_sigma, self.opt.bp_tap, self.opt.bp_sigma)
        pr_BP_BP = (pr_BP_BP - 0.5) / 0.5
        pr_BP_BP = pr_BP_BP.unsqueeze(0)
        return pr_BP_BP


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        visual_ret['pr_BP_BC'] = self.get_current_BC(self.pr_BC)
        visual_ret['pr_BP_BC2'] = self.get_current_BC(self.pr_BC2)
        visual_ret['pr_BP_BP'] = self.get_current_BP(self.pr_BP)
        visual_ret['pr_BP_BP2'] = self.get_current_BP(self.pr_BP2)
        return visual_ret

    def eval_label(self):
        label = ['idx', 'condition', 'bc_gt', 'bc_ra', 'bc_sh', 'bc_ba', 'bc_bp', 'bc_bc', 
        'bc_ba2', 'bc_bp2', 'bc_bc2', 
        'dist_ra', 'dist_sh', 'dist_ba', 'dist_bp', 'dist_bc',
        'dist_ba2', 'dist_bp2', 'dist_bc2', 'dist_05',
        'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba', 'ba_mse_ba2','ba_mse_0', 'ba_mse_h', 'ba_mse_1',
        'bp_mse_ra', 'bp_mse_sh', 'bp_mse_ba', 'bp_mse_bp', 'bp_mse_bp_direct', 
        'bp_mse_ba2', 'bp_mse_bp2', 'bp_mse_bp2_direct', 'bp_mse_0', 'bp_mse_h', 'bp_mse_1']

        return label

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        pr_SH_g = torch.squeeze(torch.mean(self.pr_SH, 1, keepdim=True), 0)*0.5+0.5
        input_g = torch.squeeze(torch.mean(self.input, 1, keepdim=True), 0)*0.5+0.5
        mask = torch.squeeze(self.mask, 0)*0.5+0.5

        pr_BA = torch.squeeze(self.pr_BA, 0)*0.5+0.5
        pr_BP = torch.squeeze(self.pr_BP, 0)*0.5+0.5
        pr_BA2 = torch.squeeze(self.pr_BA2, 0)*0.5+0.5
        pr_BP2 = torch.squeeze(self.pr_BP2, 0)*0.5+0.5

        all_one = torch.ones_like(input_g)
        all_half = torch.ones_like(input_g) * 0.5
        all_zero = torch.zeros_like(input_g)
        pr_BA_RA, _, pr_BP_RA, pr_BC_RA = util.calc_brightest(input_g, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        pr_BA_SH, _, pr_BP_SH, pr_BC_SH = util.calc_brightest(pr_SH_g, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BA, pr_BC_BA = util.calc_brightest(pr_BA, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BP, pr_BC_BP = util.calc_brightest(pr_BP, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BA2, pr_BC_BA2 = util.calc_brightest(pr_BA2, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BP2, pr_BC_BP2 = util.calc_brightest(pr_BP2, all_one, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(self.gt_BA, 0)*0.5+0.5
        ba_mse_ra = util.mse_with_mask(pr_BA_RA, gt_BA, mask).item()
        ba_mse_sh = util.mse_with_mask(pr_BA_SH, gt_BA, mask).item()
        ba_mse_ba = util.mse_with_mask(pr_BA, gt_BA, mask).item()
        ba_mse_ba2 = util.mse_with_mask(pr_BA2, gt_BA, mask).item()
        ba_mse_0 = util.mse_with_mask(all_zero, gt_BA, mask).item()
        ba_mse_h = util.mse_with_mask(all_half, gt_BA, mask).item()
        ba_mse_1 = util.mse_with_mask(all_one, gt_BA, mask).item()

        # Evaluation of brightest pixel (Spread)
        gt_BP = torch.squeeze(self.gt_BP, 0)*0.5+0.5
        bp_mse_ra = util.mse_with_mask(pr_BP_RA, gt_BP, mask).item()
        bp_mse_sh = util.mse_with_mask(pr_BP_SH, gt_BP, mask).item()
        bp_mse_ba = util.mse_with_mask(pr_BP_BA, gt_BP, mask).item()
        bp_mse_bp = util.mse_with_mask(pr_BP_BP, gt_BP, mask).item()
        bp_mse_bp_direct = util.mse_with_mask(pr_BP, gt_BP, mask).item()
        bp_mse_ba2 = util.mse_with_mask(pr_BP_BA2, gt_BP, mask).item()
        bp_mse_bp2 = util.mse_with_mask(pr_BP_BP2, gt_BP, mask).item()
        bp_mse_bp2_direct = util.mse_with_mask(pr_BP2, gt_BP, mask).item()
        bp_mse_0 = util.mse_with_mask(all_zero, gt_BP, mask).item()
        bp_mse_h = util.mse_with_mask(all_half, gt_BP, mask).item()
        bp_mse_1 = util.mse_with_mask(all_one, gt_BP, mask).item()

        # Evaluation of brightest coordinate
        bc_gt = []
        bc_gt_num = int(self.gt_BC[0, 0, 3].item())
        for i in range(bc_gt_num):
            bc_gt.append((self.gt_BC[0, i, 0].item(), self.gt_BC[0, i, 1].item(), int(self.gt_BC[0, i, 2].item()), int(self.gt_BC[0, i, 3].item())))
        bc_ra = pr_BC_RA
        bc_sh = pr_BC_SH
        bc_ba = pr_BC_BA
        bc_bp = pr_BC_BP
        bc_ba2 = pr_BC_BA2
        bc_bp2 = pr_BC_BP2
        bc_bc = [(self.pr_BC[0, 0].item(), self.pr_BC[0, 1].item(), 1, 1)]
        bc_bc2 = [(self.pr_BC2[0, 0].item(), self.pr_BC2[0, 1].item(), 1, 1)]
        bc_05 = [(0.5, 0.5, 1, 1)]

        dist_ra = util.calc_dist(bc_gt, bc_ra)
        dist_sh = util.calc_dist(bc_gt, bc_sh)
        dist_ba = util.calc_dist(bc_gt, bc_ba)
        dist_bp = util.calc_dist(bc_gt, bc_bp)
        dist_bc = util.calc_dist(bc_gt, bc_bc)
        dist_ba2 = util.calc_dist(bc_gt, bc_ba2)
        dist_bp2 = util.calc_dist(bc_gt, bc_bp2)
        dist_bc2 = util.calc_dist(bc_gt, bc_bc2)
        dist_05 = util.calc_dist(bc_gt, bc_05)

        condition = bc_gt[0][2]
        if torch.sum(mask > 0.5) < 1:
            condition = 3

        result = [condition, bc_gt[0], bc_ra[0], bc_sh[0], bc_ba[0], bc_bp[0], bc_bc[0], bc_ba2[0], bc_bp2[0], bc_bc2[0],
                     dist_ra, dist_sh, dist_ba, dist_bp, dist_bc, 
                     dist_ba2, dist_bp2, dist_bc2, dist_05,
                     ba_mse_ra, ba_mse_sh, ba_mse_ba, ba_mse_ba2, ba_mse_0, ba_mse_h, ba_mse_1,
                     bp_mse_ra, bp_mse_sh, bp_mse_ba, bp_mse_bp, bp_mse_bp_direct,
                     bp_mse_ba2, bp_mse_bp2, bp_mse_bp2_direct, bp_mse_0, bp_mse_h, bp_mse_1,
                     ]
        return result