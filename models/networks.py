import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptionsï¼Žã€€
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, last_relu=False, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_multi':
        net = MultiResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_latent':
        net = ResnetLatentGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_6blocks_enc':
        net = ResnetEncGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_lastrelu':
        net = UnetGeneratorLastRelu(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_multi':
        net = MultiUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_2decoder':
        net = UnetGenerator2Decoder(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_latent':
        net = UnetLatentGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, last_relu=last_relu)
    elif netG == 'unet_256_latent_inL':
        net = UnetLatentInLGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, last_relu=last_relu)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70Ã—70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetEncGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetDownsampleBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.Conv2d(ngf * mult, output_nc, kernel_size=1, padding=0)]
        # model += [nn.Tanh()]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class MultiResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(MultiResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_enc = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model_enc += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        model_latent = []
        for i in range(4): # 64x64 -> 4x4

            model_latent += [ResnetDownsampleBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # 4x4 -> 2x2
        model_latent += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult),
                            nn.ReLU(True)]
        # 2x2 -> 1x1
        model_latent += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            # norm_layer(ngf * mult),
                            nn.ReLU(True)]
        model_latent += [nn.Conv2d(ngf * mult, 2, kernel_size=1, padding=0)]
        # model_latent += [nn.Tanh()]
        model_latent += [nn.Sigmoid()] # 0-1


        model_dec_1 = []
        model_dec_2 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_dec_1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            model_dec_2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_dec_1 += [nn.ReflectionPad2d(3)]
        model_dec_2 += [nn.ReflectionPad2d(3)]
        model_dec_1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_dec_2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_dec_1 += [nn.Tanh()]
        model_dec_2 += [nn.Tanh()]

        self.model_enc = nn.Sequential(*model_enc)
        self.model_latent = nn.Sequential(*model_latent)
        self.model_dec_1 = nn.Sequential(*model_dec_1)
        self.model_dec_2 = nn.Sequential(*model_dec_2)

    def forward(self, input):
        """Standard forward"""
        down_x = self.model_enc(input)
        y_0 = self.model_latent(down_x)
        y_0 = torch.squeeze(torch.squeeze(y_0, 3), 2)
        y_1 = self.model_dec_1(down_x)
        y_2 = self.model_dec_2(down_x)
        return y_0, y_1, y_2

class ResnetLatentGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetLatentGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_enc = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model_enc += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        model_latent = []
        for i in range(4): # 64x64 -> 4x4

            model_latent += [ResnetDownsampleBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # 4x4 -> 2x2
        model_latent += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult),
                            nn.ReLU(True)]
        # 2x2 -> 1x1
        model_latent += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            # norm_layer(ngf * mult),
                            nn.ReLU(True)]
        model_latent += [nn.Conv2d(ngf * mult, 2, kernel_size=1, padding=0)]
        # model_latent += [nn.Tanh()]
        model_latent += [nn.Sigmoid()] # 0-1


        model_dec = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_dec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_dec += [nn.ReflectionPad2d(3)]
        model_dec += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_dec += [nn.Tanh()]

        self.model_enc = nn.Sequential(*model_enc)
        self.model_latent = nn.Sequential(*model_latent)
        self.model_dec = nn.Sequential(*model_dec)

    def forward(self, input):
        """Standard forward"""
        down_x = self.model_enc(input)
        y_0 = self.model_latent(down_x)
        y_0 = torch.squeeze(torch.squeeze(y_0, 3), 2)
        y_1 = self.model_dec(down_x)
        return y_0, y_1



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetDownsampleBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetDownsampleBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        downconv = [nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(dim),
                      nn.ReLU(True)]
        self.down = nn.Sequential(*downconv)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        x = x + self.conv_block(x)  # add skip connections
        out = self.down(x)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetGeneratorLastRelu(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorLastRelu, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, last_relu=True)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|

class MultiUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(MultiUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)
        # if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
            # self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class MultiUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MultiUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        # print("we are in mutilUnet")
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            n_output_dim = 3

            down = [downconv]

            # Shading (1ch out)
            upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
                                        kernel_size=4, stride=2,
                                        padding=1)]
            # Albedo (3ch out)
            upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 3,
                                        kernel_size=4, stride=2,
                                        padding=1)]

        elif innermost:

            down = [downrelu, downconv]
            upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            #  for rgb shading 
            int_conv = [nn.AdaptiveAvgPool2d((1,1)) , nn.ReLU(False),  nn.Conv2d(inner_nc, int(inner_nc/2), kernel_size=3, stride=1, padding=1), nn.ReLU(False)]
            fc = [nn.Linear(256, 3)]
            self.int_conv = nn.Sequential(* int_conv) 
            self.fc = nn.Sequential(* fc)
        else:

            down = [downrelu, downconv, downnorm]
            up_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            up_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model_1 = up_1 + [nn.Dropout(0.5)]
                upconv_model_2 = up_2 + [nn.Dropout(0.5)]
            else:
                upconv_model_1 = up_1
                upconv_model_2 = up_2
        

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model_1 = nn.Sequential(*upconv_model_1)
        self.upconv_model_2 = nn.Sequential(*upconv_model_2)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)

            y_1, y_2, color_s = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)

            return y_1, y_2, color_s

        elif self.innermost:
            down_output = self.downconv_model(x)
            color_s = self.int_conv(down_output)
            color_s = color_s.view(color_s.size(0), -1)
            color_s  = self.fc(color_s)

            y_1 = self.upconv_model_1(down_output)
            y_2 = self.upconv_model_2(down_output)  
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)


            return y_1, y_2, color_s
        else:
            down_x = self.downconv_model(x)
            y_1, y_2, color_s = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2, color_s


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|

class UnetLatentGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, last_relu=False, gpu_ids=[]):
        super(UnetLatentGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetLatentSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetLatentSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetLatentSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, last_relu=last_relu)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetLatentSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, last_relu=False):
        super(UnetLatentSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if input_nc is None:
            input_nc = outer_nc
        # downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
        #                      stride=2, padding=1)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            down = [downconv]

            # Shading (1ch out)
            upconv = nn.ConvTranspose2d(inner_nc * 2 , outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            # upconv_model = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2 , outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)]

            if last_relu:
                upconv_model = [uprelu, upconv, uprelu]
            else:
                upconv_model = [uprelu, upconv, nn.Tanh()]

        elif innermost:

            down = [downrelu, downconv]
            upconv_model = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            #  for rgb shading 
            int_conv = [nn.AdaptiveAvgPool2d((1,1)) , nn.ReLU(False),  nn.Conv2d(inner_nc, int(inner_nc/2), kernel_size=3, stride=1, padding=1), nn.ReLU(False)]
            fc = [nn.Linear(256, 3)]
            self.int_conv = nn.Sequential(* int_conv) 
            self.fc = nn.Sequential(* fc)
        else:

            down = [downrelu, downconv, downnorm]
            # up = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1), norm_layer(outer_nc, affine=True)]
            up = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model = up + [nn.Dropout(0.5)]
            else:
                upconv_model = up
        

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model = nn.Sequential(*upconv_model)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)

            y, color_s = self.submodule.forward(down_x)
            y = self.upconv_model(y)

            return y, color_s

        elif self.innermost:
            down_output = self.downconv_model(x)
            color_s = self.int_conv(down_output)
            color_s = color_s.view(color_s.size(0), -1)
            color_s  = self.fc(color_s)

            y = self.upconv_model(down_output)
            y = torch.cat([y, x], 1)

            return y, color_s
        else:
            down_x = self.downconv_model(x)
            y, color_s = self.submodule.forward(down_x)
            y = self.upconv_model(y)
            y = torch.cat([y, x], 1)

            return y, color_s

class UnetLatentInLGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, last_relu=False, gpu_ids=[]):
        super(UnetLatentInLGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetLatentInLSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetLatentInLSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetLatentInLSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentInLSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentInLSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetLatentInLSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, last_relu=False)

        self.model = unet_block

    def forward(self, input, L):
        return self.model(input, L)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetLatentInLSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, last_relu=False):
        super(UnetLatentInLSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if input_nc is None:
            input_nc = outer_nc
        # downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
        #                      stride=2, padding=1)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            down = [downconv]

            # Shading (1ch out)
            upconv = nn.ConvTranspose2d(inner_nc * 2 , outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            # upconv_model = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2 , outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)]

            if last_relu:
                upconv_model = [uprelu, upconv, uprelu]
            else:
                upconv_model = [uprelu, upconv, nn.Tanh()]

        elif innermost:

            down = [downrelu, downconv]
            upconv_model = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc + 128, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            #  for rgb shading 
            int_conv = [nn.AdaptiveAvgPool2d((1,1)) , nn.ReLU(False),  nn.Conv2d(inner_nc, int(inner_nc/2), kernel_size=3, stride=1, padding=1), nn.ReLU(False)]
            fc = [nn.Linear(256, 3)]
            self.int_conv = nn.Sequential(* int_conv) 
            self.fc = nn.Sequential(* fc)

            L_fc = [nn.Linear(25, 64), nn.ReLU(False),
                         nn.Linear(64, 128)
                         ]
            self.L_fc = nn.Sequential(* L_fc)
            L_up = [nn.ReLU(False), nn.ConvTranspose2d(128, 128,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(128, affine=True)]
            self.L_up = nn.Sequential(* L_up)

        else:

            down = [downrelu, downconv, downnorm]
            # up = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1), norm_layer(outer_nc, affine=True)]
            up = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model = up + [nn.Dropout(0.5)]
            else:
                upconv_model = up
        

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model = nn.Sequential(*upconv_model)

    def forward(self, x, L):

        if self.outermost:
            down_x = self.downconv_model(x)

            y, color_s = self.submodule.forward(down_x, L)
            y = self.upconv_model(y)

            return y, color_s

        elif self.innermost:
            down_output = self.downconv_model(x)
            color_s = self.int_conv(down_output)
            color_s = color_s.view(color_s.size(0), -1)
            color_s  = self.fc(color_s)

            # print('L.shape', L.shape)
            # print('down_output.shape', down_output.shape)
            Lfc = self.L_fc(L) # [bs, 128]
            Lfc = Lfc.view(Lfc.size(0), Lfc.size(1), 1, 1) # [bs, 128, 1, 1]
            Lup = self.L_up(Lfc) # [bs, 128, 2, 2]
            latent = torch.cat([down_output, Lup], 1)
            y = self.upconv_model(latent)

            # y = self.upconv_model(down_output)
            y = torch.cat([y, x], 1)

            return y, color_s
        else:
            down_x = self.downconv_model(x)
            y, color_s = self.submodule.forward(down_x, L)
            y = self.upconv_model(y)
            y = torch.cat([y, x], 1)

            return y, color_s



class UnetGenerator2Decoder(nn.Module):
    """Create a Unet-based generator (Out latent feature)"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet Latentout generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator2Decoder, self).__init__()
        # construct unet structure

        dec_num = 4
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d        
        # Down 128x128
        layer = []
        layer.append(nn.Conv2d(input_nc, ngf, kernel_size=4,stride=2, padding=1, bias=use_bias)) #128x128x64
        self.down_128 = nn.Sequential(*layer)        

        # Down 64x64
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf, ngf*2, kernel_size=4,stride=2, padding=1, bias=use_bias)) #64x64x128
        layer.append(norm_layer(ngf*2))
        self.down_64 = nn.Sequential(*layer)  
        
        # Down 32x32
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*2, ngf*4, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*4))
        self.down_32 = nn.Sequential(*layer)  

        # Down 16x16
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*4, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        self.down_16 = nn.Sequential(*layer)  

        # Down 8x8
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        self.down_8 = nn.Sequential(*layer)  

        # Down 4x4
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        self.down_4 = nn.Sequential(*layer)  

        # Down 2x2
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        self.down_2 = nn.Sequential(*layer)  

        # Down 1x1
        layer = []
        layer.append(nn.LeakyReLU(0.2, False))
        layer.append(nn.Conv2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        self.down_1 = nn.Sequential(*layer)  

        # Up 2x2
        up_2 = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*8))
            up_2.append(nn.Sequential(*layer))  
        self.up_2 = nn.ModuleList(up_2)

        # Up 4x4
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        layer.append(nn.Dropout(0.5))
        self.up_4 = nn.Sequential(*layer)  

        # Up 4x4 skip
        up_4_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*8))
            layer.append(nn.Dropout(0.5))
            up_4_cat.append(nn.Sequential(*layer))  
        self.up_4_cat = nn.ModuleList(up_4_cat)

        # Up 8x8
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        layer.append(nn.Dropout(0.5))
        self.up_8 = nn.Sequential(*layer)  

        # Up 8x8 skip
        up_8_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*8))
            layer.append(nn.Dropout(0.5))
            up_8_cat.append(nn.Sequential(*layer))  
        self.up_8_cat = nn.ModuleList(up_8_cat)

        # Up 16x16
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*8))
        layer.append(nn.Dropout(0.5))
        self.up_16 = nn.Sequential(*layer)  

        # Up 16x16 skip
        up_16_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*8))
            layer.append(nn.Dropout(0.5))
            up_16_cat.append(nn.Sequential(*layer))  
        self.up_16_cat = nn.ModuleList(up_16_cat)

        # Up 32x32
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*4))
        layer.append(nn.Dropout(0.5))
        self.up_32 = nn.Sequential(*layer)  

        # Up 32x32 skip
        up_32_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*16, ngf*4, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*4))
            layer.append(nn.Dropout(0.5))
            up_32_cat.append(nn.Sequential(*layer))  
        self.up_32_cat = nn.ModuleList(up_32_cat)
        
        # Up 64x64
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf*2))
        layer.append(nn.Dropout(0.5))
        self.up_64 = nn.Sequential(*layer)  

        # Up 64x64 skip
        up_64_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*8, ngf*2, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf*2))
            layer.append(nn.Dropout(0.5))
            up_64_cat.append(nn.Sequential(*layer))  
        self.up_64_cat = nn.ModuleList(up_64_cat)

        # Up 128x128
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(norm_layer(ngf))
        layer.append(nn.Dropout(0.5))
        self.up_128 = nn.Sequential(*layer)  

        # Up 128x128 skip
        up_128_cat = []
        for i in range(dec_num):
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*4, ngf, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(norm_layer(ngf))
            layer.append(nn.Dropout(0.5))
            up_128_cat.append(nn.Sequential(*layer))  
        self.up_128_cat = nn.ModuleList(up_128_cat)

        # Up 256x256
        layer = []
        layer.append(nn.ReLU(False))
        layer.append(nn.ConvTranspose2d(ngf, 1, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
        layer.append(nn.Tanh())
        self.up_256 = nn.Sequential(*layer)  

        # Up 256x256 skip
        up_256_cat = []
        for i in range(dec_num):
            outnc = 1 if i>=2 else 3
            layer = []
            layer.append(nn.ReLU(False))
            layer.append(nn.ConvTranspose2d(ngf*2, outnc, kernel_size=4,stride=2, padding=1, bias=use_bias)) #32x32x256
            layer.append(nn.Tanh())
            up_256_cat.append(nn.Sequential(*layer))  
        self.up_256_cat = nn.ModuleList(up_256_cat)


    def forward(self, input):
        """Standard forward"""

        # Common Encoder
        d_128 = self.down_128(input)
        d_64 = self.down_64(d_128)
        d_32 = self.down_32(d_64)
        d_16 = self.down_16(d_32)
        d_8 = self.down_8(d_16)
        d_4 = self.down_4(d_8)
        d_2 = self.down_2(d_4)
        d_1 = self.down_1(d_2)

        # Reflectance Decoder with skip connection
        R_2 = self.up_2[0](d_1)
        R_4 = self.up_4_cat[0](torch.cat([d_2, R_2], 1))
        R_8 = self.up_8_cat[0](torch.cat([d_4, R_4], 1))
        R_16 = self.up_16_cat[0](torch.cat([d_8, R_8], 1))
        R_32 = self.up_32_cat[0](torch.cat([d_16, R_16], 1))
        R_64 = self.up_64_cat[0](torch.cat([d_32, R_32], 1))
        R_128 = self.up_128_cat[0](torch.cat([d_64, R_64], 1))
        R_256 = self.up_256_cat[0](torch.cat([d_128, R_128], 1))

        # Shadowing Decoder with skip connection
        S_2 = self.up_2[1](d_1)
        S_4 = self.up_4_cat[1](torch.cat([d_2, S_2], 1))
        S_8 = self.up_8_cat[1](torch.cat([d_4, S_4], 1))
        S_16 = self.up_16_cat[1](torch.cat([d_8, S_8], 1))
        S_32 = self.up_32_cat[1](torch.cat([d_16, S_16], 1))
        S_64 = self.up_64_cat[1](torch.cat([d_32, S_32], 1))
        S_128 = self.up_128_cat[1](torch.cat([d_64, S_64], 1))
        S_256 = self.up_256_cat[1](torch.cat([d_128, S_128], 1))

        # Brightest portion Decoder
        BA_2 = self.up_2[2](d_1)
        BA_4 = self.up_4_cat[2](torch.cat([d_2, BA_2], 1))
        BA_8 = self.up_8_cat[2](torch.cat([d_4, BA_4], 1))
        BA_16 = self.up_16_cat[2](torch.cat([d_8, BA_8], 1))
        BA_32 = self.up_32_cat[2](torch.cat([d_16, BA_16], 1))
        BA_64 = self.up_64_cat[2](torch.cat([d_32, BA_32], 1))
        BA_128 = self.up_128_cat[2](torch.cat([d_64, BA_64], 1))
        BA_256 = self.up_256_cat[2](torch.cat([d_128, BA_128], 1))

        BP_2 = self.up_2[3](d_1)
        BP_4 = self.up_4_cat[3](torch.cat([d_2, BP_2], 1))
        BP_8 = self.up_8_cat[3](torch.cat([d_4, BP_4], 1))
        BP_16 = self.up_16_cat[3](torch.cat([d_8, BP_8], 1))
        BP_32 = self.up_32_cat[3](torch.cat([d_16, BP_16], 1))
        BP_64 = self.up_64_cat[3](torch.cat([d_32, BP_32], 1))
        BP_128 = self.up_128_cat[3](torch.cat([d_64, BP_64], 1))
        BP_256 = self.up_256_cat[3](torch.cat([d_128, BP_128], 1))

        output = torch.cat([R_256, S_256, BA_256, BP_256], 1)
        return output



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, last_relu=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if last_relu:
                up = [uprelu, upconv, uprelu]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class IlluminationEncoder(nn.Module):
    def __init__(self, input_nc=25, hidden_nc=10, output_nc=5, use_hidden=True):
        super(IlluminationEncoder, self).__init__()

        if use_hidden:
            fc1 = nn.Linear(input_nc, hidden_nc)
            fc2 = nn.Linear(hidden_nc, output_nc)
        else:
            fc = nn.Linear(input_nc, output_nc)

        relu = nn.ReLU(False)
        softmax = nn.Softmax()

        if use_hidden:
            self.net = nn.Sequential(fc1, relu, fc2, softmax)
        else:
            self.net = nn.Sequential(fc, softmax)

    def forward(self, input):
        return self.net(input)