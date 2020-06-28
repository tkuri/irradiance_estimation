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

class BrightestResnetModel(BaseModel):
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
        parser.add_argument('--joint_enc', action='store_true', help='joint encoder')
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
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>

        if opt.joint_enc:
            self.loss_names = ['G_AL', 'G_SH', 'G_BA', 'G_BP', 'G_BC']
        else:
            self.loss_names = ['G_AL', 'G_SH', 'G_BA', 'G_BP']

        self.visual_names = ['input', 'pr_BA', 'gt_BA', 'pr_BP', 'gt_BP', 'pr_AL', 'gt_AL', 'pr_SH', 'gt_SH', 'mask', 'mask_edge']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if opt.joint_enc:
            self.model_names = ['G1', 'G2']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2', 'G3']
        # define networks (both generator and discriminator)

        # print('generator output:', output)

        # self.netG = networks.define_G(opt.input_nc, output, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, 3, opt.ngf, 'unet_256_multi', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.joint_enc:
            self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG3 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks', opt.norm,
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
            if not opt.joint_enc:
                self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G3)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.gt_AL = torch.squeeze(input['gt_AL'],0).to(self.device) # [bn, 3, 256, 256]
        self.gt_SH = torch.squeeze(input['gt_SH'],0).to(self.device) # [bn, 3, 256, 256]
        self.mask = torch.squeeze(input['mask'],0).to(self.device) # [bn, 1, 256, 256]
        self.mask_edge = torch.squeeze(input['mask_edge'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BA = torch.squeeze(input['gt_BA'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BP = torch.squeeze(input['gt_BP'],0).to(self.device) # [bn, 1, 256, 256]
        self.gt_BC = input['gt_BC'].to(self.device) 
        self.image_paths = input['A_paths']
    
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
        if self.opt.joint_enc:
            self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)
        else:
            self.pr_BA = self.netG2(self.input)
            self.pr_BP = self.netG3(self.input)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        mask_edge = self.mask_edge*0.5 + 0.5
        gt_BC = self.gt_BC[:,:,:2]
        condition = int(self.gt_BC[:, 0, 2].item())
        bc_num = int(self.gt_BC[:, 0, 3].item())
        self.loss_G_AL = self.criterionAL(self.pr_AL*mask, self.gt_AL*mask) * self.opt.lambda_AL
        self.loss_G_SH = self.criterionSH(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_SH
        self.loss_G_BA = self.criterionBA(self.pr_BA*mask_edge, self.gt_BA*mask_edge) * self.opt.lambda_BA
        self.loss_G_BP = self.criterionBP(self.pr_BP*mask_edge, self.gt_BP*mask_edge) * self.opt.lambda_BP  

        self.loss_G = self.loss_G_AL + self.loss_G_SH + self.loss_G_BA + self.loss_G_BP
        if self.opt.joint_enc:            
#             if condition==1:
#                 self.loss_G_BC = self.criterionBC(self.pr_BC, gt_BC) * self.opt.lambda_BC
#                 self.loss_G += self.loss_G_BC
#             else:
#                 print('Pass loss_G_BC because condition is {}'.format(condition))
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
        if not self.opt.joint_enc:
            self.optimizer_G3.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights
        if not self.opt.joint_enc:
            self.optimizer_G3.step()             # udpate G's weights

    def get_current_BC(self):
        pr_BP_BC = util.disp_brightest_coord(self.pr_BC, self.mask, self.opt.bp_tap, self.opt.bp_sigma)
        pr_BP_BC = (pr_BP_BC - 0.5) / 0.5
        return pr_BP_BC

    def get_current_BP(self):
        pr_BP = torch.squeeze(self.pr_BP, 0)*0.5+0.5
        mask_edge = torch.squeeze(self.mask_edge, 0)*0.5+0.5
        _, _, pr_BP_BP, _ = util.calc_brightest(pr_BP, mask_edge, self.opt.bp_nr_tap, self.opt.bp_nr_sigma, self.opt.bp_tap, self.opt.bp_sigma)
        pr_BP_BP = (pr_BP_BP - 0.5) / 0.5
        pr_BP_BP = pr_BP_BP.unsqueeze(0)
        return pr_BP_BP

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        visual_ret['pr_BP_BC'] = self.get_current_BC()
        visual_ret['pr_BP_BP'] = self.get_current_BP()
        return visual_ret


    def eval_label(self):
        label = ['idx', 'condition', 'bc_gt', 'bc_ra', 'bc_sh', 'bc_ba', 'bc_bp', 'bc_bc', 
        'dist_ra', 'dist_sh', 'dist_ba', 'dist_bp', 'dist_bc', 'dist_05',
        'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba', 'ba_mse_0',
        'bp_mse_ra', 'bp_mse_sh', 'bp_mse_ba', 'bp_mse_bp', 'bp_mse_bp_direct', 'bp_mse_0']

        return label


    def calc_dist(self, bc_gt, bc_tar):
        dist = 10
        for i in range(int(bc_gt[0][3])):
            for j in range(int(bc_tar[0][3])):
                dist_tmp = np.hypot(bc_gt[i][0] - bc_tar[j][0], bc_gt[i][1] - bc_tar[j][1])
                if dist_tmp < dist:
                    dist = dist_tmp
        return dist

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        pr_SH_g = torch.squeeze(torch.mean(self.pr_SH, 1, keepdim=True), 0)*0.5+0.5
        input_g = torch.squeeze(torch.mean(self.input, 1, keepdim=True), 0)*0.5+0.5
        mask_edge = torch.squeeze(self.mask_edge, 0)*0.5+0.5

        pr_BA = torch.squeeze(self.pr_BA, 0)*0.5+0.5
        pr_BP = torch.squeeze(self.pr_BP, 0)*0.5+0.5

        no_mask = torch.ones_like(mask_edge)
        pr_BA_RA, _, pr_BP_RA, pr_BC_RA = util.calc_brightest(input_g, no_mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        pr_BA_SH, _, pr_BP_SH, pr_BC_SH = util.calc_brightest(pr_SH_g, no_mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BA, pr_BC_BA = util.calc_brightest(pr_BA, no_mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BP, pr_BC_BP = util.calc_brightest(pr_BP, no_mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        all_zero = torch.zeros_like(mask_edge)
        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(self.gt_BA, 0)*0.5+0.5
        ba_mse_ra = util.mse_with_mask(pr_BA_RA, gt_BA, mask_edge).item()
        ba_mse_sh = util.mse_with_mask(pr_BA_SH, gt_BA, mask_edge).item()
        ba_mse_ba = util.mse_with_mask(pr_BA, gt_BA, mask_edge).item()
        ba_mse_0 = util.mse_with_mask(all_zero, gt_BA, mask_edge).item()

        # Evaluation of brightest pixel (Spread)
        gt_BP = torch.squeeze(self.gt_BP, 0)*0.5+0.5
        bp_mse_ra = util.mse_with_mask(pr_BP_RA, gt_BP, mask_edge).item()
        bp_mse_sh = util.mse_with_mask(pr_BP_SH, gt_BP, mask_edge).item()
        bp_mse_ba = util.mse_with_mask(pr_BP_BA, gt_BP, mask_edge).item()
        bp_mse_bp = util.mse_with_mask(pr_BP_BP, gt_BP, mask_edge).item()
        bp_mse_bp_direct = util.mse_with_mask(pr_BP, gt_BP, mask_edge).item()
        bp_mse_0 = util.mse_with_mask(all_zero, gt_BP, mask_edge).item()

        # Evaluation of brightest coordinate
        bc_gt = []
        bc_gt_num = int(self.gt_BC[0, 0, 3].item())
        for i in range(bc_gt_num):
            bc_gt.append((self.gt_BC[0, i, 0].item(), self.gt_BC[0, i, 1].item(), int(self.gt_BC[0, i, 2].item()), int(self.gt_BC[0, i, 3].item())))
        bc_ra = pr_BC_RA
        bc_sh = pr_BC_SH
        bc_ba = pr_BC_BA
        bc_bp = pr_BC_BP
        bc_bc = [(self.pr_BC[0, 0].item(), self.pr_BC[0, 1].item(), 1, 1)]
        bc_05 = [(0.5, 0.5, 1, 1)]

        dist_ra = self.calc_dist(bc_gt, bc_ra)
        dist_sh = self.calc_dist(bc_gt, bc_sh)
        dist_ba = self.calc_dist(bc_gt, bc_ba)
        dist_bp = self.calc_dist(bc_gt, bc_bp)
        dist_bc = self.calc_dist(bc_gt, bc_bc)
        dist_05 = self.calc_dist(bc_gt, bc_05)

        # bc_gt = (self.gt_BC[0, 0].item(), self.gt_BC[0, 1].item(), int(self.gt_BC[0, 2].item()), int(self.gt_BC[0, 3].item()))
        # bc_ra = pr_BC_AL
        # bc_sh = pr_BC_SH
        # bc_ba = pr_BC_BA
        # bc_bp = pr_BC_BP
        # bc_bc = (self.pr_BC[0, 0].item(), self.pr_BC[0, 1].item())
        # dist_ra = np.hypot(bc_gt[0] - bc_ra[0], bc_gt[1] - bc_ra[1])
        # dist_sh = np.hypot(bc_gt[0] - bc_sh[0], bc_gt[1] - bc_sh[1])
        # dist_ba = np.hypot(bc_gt[0] - bc_ba[0], bc_gt[1] - bc_ba[1])
        # dist_bp = np.hypot(bc_gt[0] - bc_bp[0], bc_gt[1] - bc_bp[1])
        # dist_bc = np.hypot(bc_gt[0] - bc_bc[0], bc_gt[1] - bc_bc[1])
        # dist_05 = np.hypot(bc_gt[0] - 0.5, bc_gt[1] - 0.5)

        result = [bc_gt[0][2], bc_gt[0], bc_ra[0], bc_sh[0], bc_ba[0], bc_bp[0], bc_bc[0],
                     dist_ra, dist_sh, dist_ba, dist_bp, dist_bc, dist_05,
                     ba_mse_ra, ba_mse_sh, ba_mse_ba, ba_mse_0,
                     bp_mse_ra, bp_mse_sh, bp_mse_ba, bp_mse_bp, bp_mse_bp_direct, bp_mse_0                     
                     ]
        return result
