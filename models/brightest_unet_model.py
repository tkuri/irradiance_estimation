import torch
from .base_model import BaseModel
from . import networks
from typing import Union


class BrightestUnetModel(BaseModel):
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
        parser.add_argument('--netG_dec', type=int, default='1', help='The number of generator output')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for Shading loss')
            parser.add_argument('--lambda_R', type=float, default=1.0, help='weight for Reflection loss')
            parser.add_argument('--lambda_BM', type=float, default=1.0, help='weight for Brightest loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.netG_dec==1:
            self.loss_names = ['G']
            self.visual_names = ['real_I', 'fake_BM', 'real_BM', 'mask']
            output = 1
        else:
            self.loss_names = ['G_R', 'G_S', 'G_BM']
            self.visual_names = ['real_I', 'radiantest', 'fake_BM', 'real_BM', 'fake_R', 'real_R', 'fake_S', 'real_S', 'mask']
            output = opt.output_nc*2+1
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        print('generator output:', output)
    
        self.netG = networks.define_G(opt.input_nc, output, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # self.criterionR = torch.nn.L1Loss()
            # self.criterionS = torch.nn.L1Loss()
            # self.criterionBM = torch.nn.L1Loss()
            self.criterionR = torch.nn.MSELoss()
            self.criterionS = torch.nn.MSELoss()
            self.criterionBM = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

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
        self.real_BM = torch.squeeze(input['E'],0).to(self.device) # [bn, 1, 256, 256]
        self.radiantest = torch.squeeze(input['F'],0).to(self.device) # [bn, 1, 256, 256]
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
        if self.opt.netG_dec==1:
            self.fake_BM = self.netG(self.real_I)  # G(A)
        else:
            fake_RSB = self.netG(self.real_I)  # G(A)
            self.fake_R = fake_RSB[:,:self.opt.output_nc,:,:]
            self.fake_S = fake_RSB[:,self.opt.output_nc:self.opt.output_nc*2,:,:]
            self.fake_BM = fake_RSB[:,self.opt.output_nc*2:,:,:]

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        if self.opt.netG_dec==1:
            self.loss_G = self.criterionBM(self.fake_BM*mask, self.real_BM*mask) * self.opt.lambda_BM
        else:
            self.loss_G_R = self.criterionR(self.fake_R*mask, self.real_R*mask) * self.opt.lambda_R
            self.loss_G_S = self.criterionS(self.fake_S*mask, self.real_S*mask) * self.opt.lambda_S
            self.loss_G_BM = self.criterionBM(self.fake_BM*mask, self.real_BM*mask) * self.opt.lambda_BM
            self.loss_G = self.loss_G_R + self.loss_G_S + self.loss_G_BM
        self.loss_G.backward()

    def optimize_parameters(self):
        # with torch.autograd.set_detect_anomaly(True):
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
