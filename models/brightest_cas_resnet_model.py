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
import cv2

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
            parser.add_argument('--reg', action='store_true', help='regularization')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_AL', 'G_SH', 'G_BA', 'G_BP', 'G_BC']
        self.visual_names = ['input', 'pr_BA', 'pr_BA2', 'gt_BA', 'pr_BP', 'pr_BP2', 'gt_BP', 'pr_AL', 'gt_AL', 'pr_SH', 'gt_SH', 'mask', 'mask_edge']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G1', 'G2', 'G3']
        # define networks (both generator and discriminator)

        # print('generator output:', output)

        # self.netG = networks.define_G(opt.input_nc, output, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, 3, opt.ngf, 'unet_256_multi', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, 1, opt.ngf, 'resnet_9blocks_multi', opt.norm,
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
        self.pr_BC, self.pr_BA, self.pr_BP = self.netG2(self.input)
        self.pr_BC2, self.pr_BA2, self.pr_BP2 = self.netG3(self.pr_SH)
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        mask = self.mask*0.5 + 0.5
        mask_edge = self.mask_edge*0.5 + 0.5
        gt_BC = self.gt_BC[:, :2]
        condition = self.gt_BC[:, 2]
        self.loss_G_AL = self.criterionR(self.pr_AL*mask, self.gt_AL*mask) * self.opt.lambda_AL
        self.loss_G_SH = self.criterionS(self.pr_SH*mask, self.gt_SH*mask) * self.opt.lambda_S
        self.loss_G_BA = self.criterionBA(self.pr_BA*mask_edge, self.gt_BA*mask_edge) * self.opt.lambda_BA
        self.loss_G_BP = self.criterionBP(self.pr_BP*mask_edge, self.gt_BP*mask_edge) * self.opt.lambda_BP  
        self.loss_G_BC = self.criterionBC(self.pr_BC, gt_BC) * self.opt.lambda_BC
        self.loss_G_BA2 = self.criterionBA(self.pr_BA2*mask_edge, self.gt_BA*mask_edge) * self.opt.lambda_BA
        self.loss_G_BP2 = self.criterionBP(self.pr_BP2*mask_edge, self.gt_BP*mask_edge) * self.opt.lambda_BP  
        self.loss_G_BC2 = self.criterionBC(self.pr_BC2, gt_BC) * self.opt.lambda_BC

        self.loss_G = self.loss_G_AL + self.loss_G_SH + self.loss_G_BA + self.loss_G_BP + self.loss_G_BA2 + self.loss_G_BP2
        if condition==1:
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
        pr_BP_BC = util.disp_brightest_coord(pr_BC, self.mask, self.opt.bp_tap, self.opt.bp_sigma)
        pr_BP_BC = (pr_BP_BC - 0.5) / 0.5
        return pr_BP_BC

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        visual_ret['pr_BP_BC'] = self.get_current_BC(self.pr_BC)
        visual_ret['pr_BP_BC2'] = self.get_current_BC(self.pr_BC2)
        return visual_ret

    def eval_label(self):
        label = ['idx', 'condition', 'bc_gt', 'bc_ra', 'bc_sh', 'bc_ba', 'bc_bp', 'bc_bc', 
        'bc_ba2', 'bc_bp2', 'bc_bc2', 
        'dist_ra', 'dist_sh', 'dist_ba', 'dist_bp', 'dist_bc',
        'dist_ba2', 'dist_bp2', 'dist_bc2', 'dist_05',
        'ba_mse_ra', 'ba_mse_sh', 'ba_mse_ba', 'ba_mse_ba2',
        'bp_mse_ra', 'bp_mse_sh', 'bp_mse_ba', 'bp_mse_bp', 'bp_mse_ba2', 'bp_mse_bp2']

        return label

    def eval_brightest_pixel(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()
        pr_SH_g = torch.squeeze(torch.mean(self.pr_SH, 1, keepdim=True), 0)*0.5+0.5
        input_g = torch.squeeze(torch.mean(self.input, 1, keepdim=True), 0)*0.5+0.5
        mask_edge = torch.squeeze(self.mask_edge, 0)*0.5+0.5

        pr_BA = torch.squeeze(self.pr_BA, 0)*0.5+0.5
        pr_BP = torch.squeeze(self.pr_BP, 0)*0.5+0.5
        pr_BA2 = torch.squeeze(self.pr_BA2, 0)*0.5+0.5
        pr_BP2 = torch.squeeze(self.pr_BP2, 0)*0.5+0.5

        pr_BA_AL, _, pr_BP_AL, pr_BC_AL = util.calc_brightest(input_g, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        pr_BA_SH, _, pr_BP_SH, pr_BC_SH = util.calc_brightest(pr_SH_g, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BA, pr_BC_BA = util.calc_brightest(pr_BA, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, _, pr_BC_BP = util.calc_brightest(pr_BP, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, pr_BP_BA2, pr_BC_BA2 = util.calc_brightest(pr_BA2, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
        _, _, _, pr_BC_BP2 = util.calc_brightest(pr_BP2, mask_edge, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(self.gt_BA, 0)*0.5+0.5
        ba_mse_ra = util.mse_with_mask(pr_BA_AL, gt_BA, mask_edge).item()
        ba_mse_sh = util.mse_with_mask(pr_BA_SH, gt_BA, mask_edge).item()
        ba_mse_ba = util.mse_with_mask(pr_BA, gt_BA, mask_edge).item()
        ba_mse_ba2 = util.mse_with_mask(pr_BA2, gt_BA, mask_edge).item()

        # Evaluation of brightest pixel (Spread)
        gt_BP = torch.squeeze(self.gt_BP, 0)*0.5+0.5
        bp_mse_ra = util.mse_with_mask(pr_BP_AL, gt_BP, mask_edge).item()
        bp_mse_sh = util.mse_with_mask(pr_BP_SH, gt_BP, mask_edge).item()
        bp_mse_ba = util.mse_with_mask(pr_BP_BA, gt_BP, mask_edge).item()
        bp_mse_bp = util.mse_with_mask(pr_BP, gt_BP, mask_edge).item()
        bp_mse_ba2 = util.mse_with_mask(pr_BP_BA2, gt_BP, mask_edge).item()
        bp_mse_bp2 = util.mse_with_mask(pr_BP2, gt_BP, mask_edge).item()

        # Evaluation of brightest coordinate
        bc_gt = (self.gt_BC[0, 0].item(), self.gt_BC[0, 1].item(), int(self.gt_BC[0, 2].item()), int(self.gt_BC[0, 3].item()))
        bc_ra = pr_BC_AL
        bc_sh = pr_BC_SH
        bc_ba = pr_BC_BA
        bc_bp = pr_BC_BP
        bc_ba2 = pr_BC_BA2
        bc_bp2 = pr_BC_BP2
        bc_bc = (self.pr_BC[0, 0].item(), self.pr_BC[0, 1].item())
        bc_bc2 = (self.pr_BC2[0, 0].item(), self.pr_BC2[0, 1].item())
        dist_ra = np.hypot(bc_gt[0] - bc_ra[0], bc_gt[1] - bc_ra[1])
        dist_sh = np.hypot(bc_gt[0] - bc_sh[0], bc_gt[1] - bc_sh[1])
        dist_ba = np.hypot(bc_gt[0] - bc_ba[0], bc_gt[1] - bc_ba[1])
        dist_bp = np.hypot(bc_gt[0] - bc_bp[0], bc_gt[1] - bc_bp[1])
        dist_bc = np.hypot(bc_gt[0] - bc_bc[0], bc_gt[1] - bc_bc[1])
        dist_ba2 = np.hypot(bc_gt[0] - bc_ba2[0], bc_gt[1] - bc_ba2[1])
        dist_bp2 = np.hypot(bc_gt[0] - bc_bp2[0], bc_gt[1] - bc_bp2[1])
        dist_bc2 = np.hypot(bc_gt[0] - bc_bc2[0], bc_gt[1] - bc_bc2[1])
        dist_05 = np.hypot(bc_gt[0] - 0.5, bc_gt[1] - 0.5)

        result = [bc_gt[2], bc_gt, bc_ra, bc_sh, bc_ba, bc_bp, bc_bc, bc_ba2, bc_bp2, bc_bc2,
                     dist_ra, dist_sh, dist_ba, dist_bp, dist_bc, 
                     dist_ba2, dist_bp2, dist_bc2, dist_05,
                     ba_mse_ra, ba_mse_sh, ba_mse_ba, ba_mse_ba2,
                     bp_mse_ra, bp_mse_sh, bp_mse_ba, bp_mse_bp, bp_mse_ba2, bp_mse_bp2                      
                     ]
        return result

    def compute_pr(self, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, thres_count=400):

        thres_list = saw_utils.gen_pr_thres_list(thres_count)
        photo_ids = saw_utils.load_photo_ids_for_split(
            splits_dir=splits_dir, dataset_split=dataset_split)

        plot_arrs = []
        line_names = []

        fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
        title = '%s Precision-Recall' % (
            {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
        )

        print("FN ", fn)
        print("title ", title)

        # compute PR 
        rdic_list = self.get_precision_recall_list_new(pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size)

        plot_arr = np.empty((len(rdic_list) + 2, 2))

        # extrapolate starting point 
        plot_arr[0, 0] = 0.0
        plot_arr[0, 1] = rdic_list[0]['overall_prec']

        for i, rdic in enumerate(rdic_list):
            plot_arr[i+1, 0] = rdic['overall_recall']
            plot_arr[i+1, 1] = rdic['overall_prec']

        # extrapolate end point
        plot_arr[-1, 0] = 1
        plot_arr[-1, 1] = 0.5

        AP = np.trapz(plot_arr[:,1], plot_arr[:,0])

        return AP


    def get_precision_recall_list_new(self, pixel_labels_dir, thres_list, photo_ids,
                                  class_weights, bl_filter_size):

        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)
            for _ in range(output_count)
        ]

        count = 0 
        eval_num = 1
        # total_num_img = len(photo_ids)
        total_num_img = eval_num

        # for photo_id in (photo_ids):
        for photo_id in photo_ids[:total_num_img]:
            print("photo_id ", count, photo_id, total_num_img)

            saw_img_ori = saw_utils.load_img_arr(photo_id)
            original_h, original_w = saw_img_ori.shape[0], saw_img_ori.shape[1]
            saw_img = saw_utils.resize_img_arr(saw_img_ori)

            saw_img = np.transpose(saw_img, (2,0,1))
            input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
            input_images = input_
            # input_images = Variable(input_.cuda() , requires_grad = False)

            # prediction_S, prediction_R, rgb_s = self.netG.forward(input_images) 
            prediction_S, prediction_A, _ = self.netG1(input_images)
            prediction_S = util.normalize_n1p1_to_0p1(grayscale=True)(prediction_S.data[0,:,:,:])
            prediction_A = util.normalize_n1p1_to_0p1(grayscale=False)(prediction_A.data[0,:,:,:])
            
            # prediction_Sr = torch.exp(prediction_S)
            # prediction_S_np = prediction_S.data[0,0,:,:].cpu().numpy()
            prediction_S_np = prediction_S.data[0,:,:].cpu().numpy()
            prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)
            prediction_A_np = prediction_A.data[:,:,:].cpu().numpy()
            prediction_A_np = np.transpose(prediction_A_np, (1, 2, 0))
            prediction_A_np = resize(prediction_A_np, (original_h, original_w), order=1, preserve_range=True)
            cv2.imwrite('test_shading.png', prediction_S_np*255)
            cv2.imwrite('test_albedo.png', cv2.cvtColor((prediction_A_np*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite('test_original.png', cv2.cvtColor((saw_img_ori*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            

            # compute confusion matrix
            conf_mx_list = self.eval_on_images( shading_image_arr = prediction_S_np,
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_id=photo_id, bl_filter_size = bl_filter_size
            )


            # if np.all(conf_mx_list == 0) == False:
            #     print("no label")
            #     continue

            for i, conf_mx in enumerate(conf_mx_list):
                # If this image didn't have any labels
                if conf_mx is None:
                    continue

                overall_conf_mx_list[i] += conf_mx

            count += 1

        ret = []
        for i in range(output_count):
            overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
            )

            ret.append(dict(
                overall_prec=overall_prec,
                overall_recall=overall_recall,
                overall_conf_mx=overall_conf_mx_list[i],
            ))

        return ret


    def eval_on_images(self, shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size):
        """
        This method generates a list of precision-recall pairs and confusion
        matrices for each threshold provided in ``thres_list`` for a specific
        photo.

        :param shading_image_arr: predicted shading images

        :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

        :param thres_list: List of shading gradient magnitude thresholds we use to
        generate points on the precision-recall curve.

        :param photo_id: ID of the photo we want to evaluate on.

        :param bl_filter_size: The size of the maximum filter used on the shading
        gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
        """

        shading_image_grayscale = shading_image_arr
        shading_image_grayscale[shading_image_grayscale < 1e-4] = 1e-4
        shading_image_grayscale = np.log(shading_image_grayscale)

        shading_gradmag = saw_utils.compute_gradmag(shading_image_grayscale)
        shading_gradmag = np.abs(shading_gradmag)

        if bl_filter_size:
            shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

        # We have the following ground truth labels:
        # (0) normal/depth discontinuity non-smooth shading (NS-ND)
        # (1) shadow boundary non-smooth shading (NS-SB)
        # (2) smooth shading (S)
        # (100) no data, ignored
        y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
        
        y_true = np.ravel(y_true)
        ignored_mask = y_true == 100

        # If we don't have labels for this photo (so everything is ignored), return
        # None
        if np.all(ignored_mask):
            print("no labels")
            return [None] * len(thres_list)

        ret = []
        for thres in thres_list:
            y_pred = (shading_gradmag < thres).astype(int)
            y_pred_max = (shading_gradmag_max < thres).astype(int)
            y_pred = np.ravel(y_pred)
            y_pred_max = np.ravel(y_pred_max)
            # Note: y_pred should have the same image resolution as y_true
            assert y_pred.shape == y_true.shape

            confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            ret.append(confusion_matrix)

        return ret

