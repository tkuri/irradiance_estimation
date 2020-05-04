import torch
from .base_model import BaseModel
from . import networks
from torch.nn import functional as F

class Pix2PixTm2In2MultiModel(BaseModel):
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

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_C']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_C', 'real_C_itp2', 'matrix_1', 'matrix_2']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'G2', 'D']
        else:  # during test time, only load G
            self.model_names = ['G', 'G2']

        # define networks (both generator and discriminator)
        self.output_nc = opt.output_nc
        self.light_res = opt.light_res
        print('opt.output_nc', opt.output_nc)
        print('light_res', self.light_res)

        self.netG = networks.define_G(opt.input_nc + opt.input2_nc, opt.output_nc, opt.ngf, 'unet_256_lastrelu', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                                      
        self.netG2 = networks.define_G(opt.input_nc + opt.input2_nc, 1, opt.ngf, 'unet_256_lastrelu', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.input2_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)
            


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = torch.squeeze(input['A'],0).to(self.device) # [25, 3, 256, 256]
        self.real_B = torch.squeeze(input['B'],0).to(self.device) # [25, 3, 256, 256]
        self.real_C = torch.squeeze(input['C'],0).to(self.device) # [25, 1, 256, 256]
        self.real_C_itp = F.interpolate(self.real_C, (self.light_res, self.light_res), mode='bilinear', align_corners=False)
        self.real_C_itp_flat = self.real_C_itp.view(-1, self.light_res**2, 1) # [1, lsxls, 1]
        self.real_C_itp2 = torch.clamp((F.interpolate(self.real_C_itp, (self.real_C.size(-2), self.real_C.size(-1)), mode='nearest')-0.5)/0.5, min=-1.0, max=1.0)
        self.real_AC = torch.cat([self.real_A, self.real_C], dim=1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        # self.matrix_1_gain = 0.25
        # self.matrix_2_gain = 64.0
        self.matrix_1_gain = 1.0
        self.matrix_2_gain = 1.0

    def forward(self):
        # print("test")
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        sub_matrix1 = self.netG(self.real_AC) # [25, 3, 256, 256]
        sub_matrix2 = self.netG2(self.real_AC) # [25, 1, 256, 256]
        sub_matrix2 = F.interpolate(sub_matrix2, (self.light_res, self.light_res), mode='bilinear', align_corners=False)# [25, ls, ls]
        
        self.matrix_1 = torch.clamp((sub_matrix1.clone()*self.matrix_1_gain-0.5)/0.5, min=-1.0, max=1.0)
        self.matrix_2 = torch.clamp((F.interpolate(sub_matrix2, (self.real_B.size(-2), self.real_B.size(-1)), mode='nearest')*self.matrix_2_gain-0.5)/0.5, min=-1.0, max=1.0)
        
        sub_matrix1 = sub_matrix1.view(-1, sub_matrix1.size(1)*sub_matrix1.size(2)*sub_matrix1.size(3), 1) # [25, 3x256x256, 1]
        sub_matrix2 = sub_matrix2.view(-1, 1, sub_matrix2.size(-2)*sub_matrix2.size(-1)) # [25, 1, lsxls]
        sub_matrix3 = torch.matmul(sub_matrix2, self.real_C_itp_flat) #[1, 1, 1]

        tmR = sub_matrix1[:, 0:256**2, :] # [25, 256x256, lsxls]
        tmG = sub_matrix1[:, 256**2:(256**2)*2, :]
        tmB = sub_matrix1[:, (256**2)*2:(256**2)*3, :]
        bufR = torch.matmul(tmR, sub_matrix3) # [25, 256x256, 1]
        bufG = torch.matmul(tmG, sub_matrix3)
        bufB = torch.matmul(tmB, sub_matrix3)
        buf = torch.cat([bufR, bufG, bufB], dim=2) # [25, 256x256, 3]
        buf = torch.transpose(buf, 1, 2) # [25, 3, 256x256]
        buf = (buf - 0.5) / 0.5
        buf = torch.clamp(buf, min=-1.0, max=1.0)
        self.fake_B = buf.view(self.real_B.size()) # [25, 3, 256, 256]

    def forward_linebuf(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        sub_matrix1 = self.netG(self.real_AC) # [25, 3, 256, 256]
        sub_matrix2 = self.netG2(self.real_AC) # [25, 1, 256, 256]
        sub_matrix2 = F.interpolate(sub_matrix2, (self.light_res, self.light_res), mode='bilinear', align_corners=False)
        self.fake_B = torch.zeros_like(self.real_B)
        sub_matrix2 = sub_matrix2.view(-1, 1, sub_matrix2.size(-2)*sub_matrix2.size(-1)) * 0.5 + 0.5 # [25, 1, 256x256]
        
        for l in range(sub_matrix1.size(2)):
            sub_matrix1_buf = sub_matrix1[:, :, l, :].reshape(-1, sub_matrix1.size(1)*sub_matrix1.size(3), 1) * 0.5 + 0.5 # [25, 3x256, 1]
            trans_matrix = torch.matmul(sub_matrix1_buf, sub_matrix2) #[25, 3x256, 256x256]
            # print('trans_matrix:', trans_matrix.size())
            tmR = trans_matrix[:, 0:256, :] # [25, 256, 256x256]
            tmG = trans_matrix[:, 256:256*2, :]
            tmB = trans_matrix[:, 256*2:256*3, :]
            # print('self.real_C_itp_flat:', self.real_C_itp_flat.size())
            # print('tmR:', tmR.size())
            bufR = torch.matmul(tmR, self.real_C_itp_flat * 10.0) # [25, 256, 1]
            bufG = torch.matmul(tmG, self.real_C_itp_flat * 10.0)
            bufB = torch.matmul(tmB, self.real_C_itp_flat * 10.0)
            # print('bufR:', bufR.size())
            buf = torch.cat([bufR, bufG, bufB], dim=2) # [25, 256, 3]
            buf = torch.transpose(buf, 1, 2) # [25, 3, 256]
            buf = (buf - 0.5) / 0.5
            buf = buf.reshape(self.fake_B.size(0), self.fake_B.size(1), self.fake_B.size(3))
            # print('buf:', buf.size())
            # print('fake_B:', self.fake_B.size())
            self.fake_B[:, :, l, :] = buf # [25, 3, 1, 256] <- [25, 3, 256]


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_ACB = torch.cat((self.real_AC, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_ACB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_ACB = torch.cat((self.real_AC, self.real_B), 1)
        pred_real = self.netD(real_ACB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_ACB = torch.cat((self.real_AC, self.fake_B), 1)
        pred_fake = self.netD(fake_ACB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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
        # self.optimizer_G.zero_grad()        # set G's gradients to zero
        # self.backward_G()                   # calculate graidents for G
        # self.optimizer_G.step()             # udpate G's weights

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights
