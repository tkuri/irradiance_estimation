import torch
from .base_model import BaseModel
from . import networks
from typing import Union
from util import util

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

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_R', 'G_S', 'G_BA', 'G_BP']
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
        self.real_BC = input['G']
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
            self.fake_BA, self.fake_BP = self.netG2(self.real_I)
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
        self.loss_G = self.loss_G_R + self.loss_G_S + self.loss_G_BA + self.loss_G_BP
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
        fake_S_gray = torch.mean(self.fake_S, 1, keepdim=True)
        real_I_gray = torch.mean(self.real_I, 1, keepdim=True)        
        _, fake_BC_BrightestPixel = util.calc_brightest_pixel(torch.squeeze(self.fake_BP,0), apply_blur=False)
        _, fake_BC_BrightestArea = util.calc_brightest_pixel(torch.squeeze(self.fake_BA,0), apply_blur=False)
        fake_BA_Shading, _ = util.calc_brightest_area(torch.squeeze(fake_S_gray,0), torch.squeeze(self.mask,0))
        _, fake_BC_Shading = util.calc_brightest_pixel(fake_BA_Shading, gauss_sigma=self.opt.brightest_sigma, apply_blur=True)
        fake_BA_Radiance, _ = util.calc_brightest_area(torch.squeeze(real_I_gray,0), torch.squeeze(self.mask,0))
        _, fake_BC_Radiance = util.calc_brightest_pixel(fake_BA_Radiance, gauss_sigma=self.opt.brightest_sigma, apply_blur=True)
        (gt_x, gt_y) = (self.real_BC[0].item(), self.real_BC[1].item())
        (ra_x, ra_y) = fake_BC_Radiance
        (sh_x, sh_y) = fake_BC_Shading
        (ba_x, ba_y) = fake_BC_BrightestArea
        (bp_x, bp_y) = fake_BC_BrightestPixel
        dist_ra = ((gt_x - ra_x)**2 + (gt_y - ra_y)**2)**0.5
        dist_sh = ((gt_x - sh_x)**2 + (gt_y - sh_y)**2)**0.5
        dist_ba = ((gt_x - ba_x)**2 + (gt_y - ba_y)**2)**0.5
        dist_bp = ((gt_x - bp_x)**2 + (gt_y - bp_y)**2)**0.5

        result = [(gt_x, gt_y), (ra_x, ra_y), (sh_x, sh_y), (ba_x, ba_y), (bp_x, bp_y), dist_ra, dist_sh, dist_ba, dist_bp]
        return result
