import torch
from .base_model import BaseModel
from . import networks
from typing import Union
from util import util
import numpy as np

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
            parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for Shading loss')
            parser.add_argument('--lambda_R', type=float, default=1.0, help='weight for Reflection loss')
            parser.add_argument('--lambda_BA', type=float, default=1.0, help='weight for Brightest area loss')
            parser.add_argument('--lambda_BP', type=float, default=1.0, help='weight for Brightest pixel loss')
            parser.add_argument('--lambda_BC', type=float, default=1.0, help='weight for Brightest coordinate loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_R', 'G_S', 'G_BA', 'G_BP', 'G_BC']
        self.visual_names = ['real_I', 'fake_BA', 'real_BA', 'fake_BP', 'real_BP', 'fake_R', 'real_R', 'fake_S', 'real_S', 'mask']
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
            self.criterionR = torch.nn.MSELoss()
            self.criterionS = torch.nn.MSELoss()
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
        self.real_I = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.real_R = torch.squeeze(input['B'],0).to(self.device) # [bn, 3, 256, 256]
        self.real_S = torch.squeeze(input['C'],0).to(self.device) # [bn, 3, 256, 256]
        self.mask = torch.squeeze(input['D'],0).to(self.device) # [bn, 1, 256, 256]
        self.real_BA = torch.squeeze(input['E'],0).to(self.device) # [bn, 1, 256, 256]
        self.real_BP = torch.squeeze(input['F'],0).to(self.device) # [bn, 1, 256, 256]
        self.real_BC = input['G'].to(self.device) 
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
        fake_S, fake_R, color = self.netG1(self.real_I)  # G(A)
        self.fake_R = fake_R
        # fake_S = fake_S.repeat(fake_S.size(0), 3, fake_S.size(2), fake_S.size(3))
        fake_S = fake_S.repeat(1, 3, 1, 1)
        color = torch.unsqueeze(torch.unsqueeze(color, 2), 3)
        self.fake_S = fake_S * color
        if self.opt.joint_enc:
            self.fake_BC, self.fake_BA, self.fake_BP = self.netG2(self.real_I)
        else:
            self.fake_BA = self.netG2(self.real_I)
            self.fake_BP = self.netG3(self.real_I)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        self.loss_G_R = self.criterionR(self.fake_R*mask, self.real_R*mask) * self.opt.lambda_R
        self.loss_G_S = self.criterionS(self.fake_S*mask, self.real_S*mask) * self.opt.lambda_S
        self.loss_G_BA = self.criterionBA(self.fake_BA*mask, self.real_BA*mask) * self.opt.lambda_BA
        self.loss_G_BP = self.criterionBP(self.fake_BP*mask, self.real_BP*mask) * self.opt.lambda_BP
        self.loss_G_BC = self.criterionBC(self.fake_BC, self.real_BC) * self.opt.lambda_BC
        self.loss_G = self.loss_G_R + self.loss_G_S + self.loss_G_BA + self.loss_G_BP + self.loss_G_BC
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

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        fake_S_g = torch.squeeze(torch.mean(self.fake_S, 1, keepdim=True), 0)
        real_I_g = torch.squeeze(torch.mean(self.real_I, 1, keepdim=True), 0)
        mask = torch.squeeze(self.mask, 0)
        fake_BA = torch.squeeze(self.fake_BA, 0)
        fake_BP = torch.squeeze(self.fake_BP, 0)

        fake_BA_R, _, fake_BP_R, fake_BC_R = util.calc_brightest(real_I_g, mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        fake_BA_S, _, fake_BP_S, fake_BC_S = util.calc_brightest(fake_S_g, mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, fake_BP_BA, fake_BC_BA = util.calc_brightest(fake_BA, mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, _, fake_BC_BP = util.calc_brightest(fake_BP, mask, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        real_BA = torch.squeeze(self.real_BA, 0)
        ba_mse_ra = util.mse_with_mask(fake_BA_R, real_BA, mask).item()
        ba_mse_sh = util.mse_with_mask(fake_BA_S, real_BA, mask).item()
        ba_mse_ba = util.mse_with_mask(fake_BA, real_BA, mask).item()

        # Evaluation of brightest pixel (Spread)
        real_BP = torch.squeeze(self.real_BP, 0)
        bp_mse_ra = util.mse_with_mask(fake_BP_R, real_BP, mask).item()
        bp_mse_sh = util.mse_with_mask(fake_BP_S, real_BP, mask).item()
        bp_mse_ba = util.mse_with_mask(fake_BP_BA, real_BP, mask).item()
        bp_mse_bp = util.mse_with_mask(fake_BP, real_BP, mask).item()

        # Evaluation of brightest coordinate
        bc_gt = (self.real_BC[0, 0].item(), self.real_BC[0, 1].item(), int(self.real_BC[0, 2].item()), int(self.real_BC[0, 3].item()))
        bc_ra = fake_BC_R
        bc_sh = fake_BC_S
        bc_ba = fake_BC_BA
        bc_bp = fake_BC_BP
        bc_bc = (self.fake_BC[0, 0].item(), self.fake_BC[0, 1].item())
        dist_ra = np.hypot(bc_gt[0] - bc_ra[0], bc_gt[1] - bc_ra[1])
        dist_sh = np.hypot(bc_gt[0] - bc_sh[0], bc_gt[1] - bc_sh[1])
        dist_ba = np.hypot(bc_gt[0] - bc_ba[0], bc_gt[1] - bc_ba[1])
        dist_bp = np.hypot(bc_gt[0] - bc_bp[0], bc_gt[1] - bc_bp[1])
        dist_bc = np.hypot(bc_gt[0] - bc_bc[0], bc_gt[1] - bc_bc[1])
        dist_05 = np.hypot(bc_gt[0] - 0.5, bc_gt[1] - 0.5)

        result = [bc_gt, bc_ra, bc_sh, bc_ba, bc_bp, bc_bc,
                     dist_ra, dist_sh, dist_ba, dist_bp, dist_bc, dist_05,
                     ba_mse_ra, ba_mse_sh, ba_mse_ba,
                     bp_mse_ra, bp_mse_sh, bp_mse_ba, bp_mse_bp                     
                     ]
        return result
