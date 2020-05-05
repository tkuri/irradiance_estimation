import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from torch.nn import functional as F

class Pix2PixTmRegModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned3')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_LTMReg', type=float, default=100.0, help='weight for LTM Regularization')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_LTMReg', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_C', 'real_C_itp', 'ltm_slice00', 'ltm_slice12', 'ltm_slice24']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.output_nc = opt.output_nc
        self.light_res = opt.light_res
        self.G_input = opt.G_input
        self.D_input = opt.D_input

        print('opt.output_nc', opt.output_nc)
        print('light_res', self.light_res)
        print('G_input', self.G_input)
        print('D_input', self.D_input)

        G_input = opt.input_nc if self.G_input=='A' else opt.input_nc + opt.input2_nc
        D_input = opt.input_nc if self.D_input=='A' else opt.input_nc + opt.input2_nc
        self.netG = networks.define_G(G_input, (self.light_res**2)*opt.output_nc, opt.ngf, 'unet_256_lastrelu', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(D_input + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            

        self.light_gain = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.tanH = nn.Tanh()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = torch.squeeze(input['A'],0).to(self.device) # [bn, 3, 256, 256]
        self.real_B = torch.squeeze(input['B'],0).to(self.device) # [bn, 3, 256, 256]
        self.real_C = torch.squeeze(input['C'],0).to(self.device) # [bn, 3, 256, 256]
        
        self.real_C_itp = F.interpolate(self.real_C, (self.light_res, self.light_res), mode='bilinear', align_corners=False) # [bn, 1, 5, 5]
        self.real_C_itp_flat = self.real_C_itp.view(-1, self.light_res**2, 1) # [bn, 25, 1]
        self.real_C_itp = torch.clamp((F.interpolate(self.real_C_itp, (self.real_C.size(-2), self.real_C.size(-1)), mode='nearest')-0.5)/0.5, min=-1.0, max=1.0)  # [bn, 256, 256, 1]
        self.real_AC = torch.cat([self.real_A, self.real_C], dim=1) # [25, 4, 256, 256]
        self.image_paths = input['A_paths']
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        src = self.real_A if self.G_input=='A' else self.real_AC
        self.trans_matrix = self.netG(src) # [25, 3*16, 256, 256]
        self.ltm_slice00 = torch.clamp((self.trans_matrix[:, [0, 25, 25*2], :, :] - 0.5) / 0.5, min=-1.0, max=1.0) # [bn, 3, 256, 256]
        self.ltm_slice12 = torch.clamp((self.trans_matrix[:, [12, 25+12, 25*2+12], :, :] - 0.5) / 0.5, min=-1.0, max=1.0) # [bn, 3, 256, 256]
        self.ltm_slice24 = torch.clamp((self.trans_matrix[:, [24, 25+24, 25*2+24], :, :] - 0.5) / 0.5, min=-1.0, max=1.0) # [bn, 3, 256, 256]
        trans_matrix = self.trans_matrix.view(-1, self.output_nc*self.light_res**2, (self.trans_matrix.size(-1)*self.trans_matrix.size(-2)))  # [bn, 3*25, 256x256]
        trans_matrix = torch.transpose(trans_matrix, 1, 2)  # [bn, 256x256, 3*25]
        tmR = trans_matrix[:, :, 0:self.light_res**2] # [bn, 256x256, 25]
        tmG = trans_matrix[:, :, self.light_res**2:(self.light_res**2)*2]
        tmB = trans_matrix[:, :, (self.light_res**2)*2:(self.light_res**2)*3]
        bufR = torch.matmul(tmR, self.real_C_itp_flat) # [bn, 256x256, 1]
        bufG = torch.matmul(tmG, self.real_C_itp_flat)
        bufB = torch.matmul(tmB, self.real_C_itp_flat)
        buf = torch.cat([bufR, bufG, bufB], dim=2) # [bn, 256x256, 3]
        buf = torch.transpose(buf, 1, 2) # [bn, 3, 256x256]
        buf = (buf - 0.5) / 0.5
        buf = torch.clamp(buf, min=-1.0, max=1.0)
        self.fake_B = buf.view(self.real_B.size()) # [bn, 3, 256, 256]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        src = self.real_A if self.D_input=='A' else self.real_AC
        fake_AB = torch.cat((src, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((src, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        src = self.real_A if self.D_input=='A' else self.real_AC
        fake_AB = torch.cat((src, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # Third, LTM Regularization
        trans_mean = torch.mean(self.trans_matrix, dim=0, keepdim=True) # [1, 75, 256, 256]
        trans_mean = trans_mean.expand(self.trans_matrix.size(0), trans_mean.size(1), trans_mean.size(2), trans_mean.size(3))  # [25, 75, 256, 256]
        self.loss_G_LTMReg = self.criterionL1(self.trans_matrix, trans_mean) * self.opt.lambda_LTMReg

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_LTMReg
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
