import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from util import util
import numpy as np
from . import saw_utils
from skimage.transform import resize
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
import cv2
import json
from skimage.measure import compare_ssim, compare_psnr

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval_bp_sh(self, mask, gt_BA, gt_BP, gt_BC, pr_SH):
        result = {}

        mask = torch.squeeze(mask, 0)*0.5+0.5
        all_one = torch.ones_like(mask)
        if self.opt.eval_mask_calc_bp:
            mask_bp = mask
        else:
            mask_bp = all_one

        pr_SH_g = torch.squeeze(torch.mean(pr_SH, 1, keepdim=True), 0)*0.5+0.5
        pr_BA_SH, _, pr_BP_SH, pr_BC_SH = util.calc_brightest(pr_SH_g, mask_bp, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(gt_BA, 0)*0.5+0.5
        result['baMSE_pr_BA_SH'] = util.mse_with_mask(pr_BA_SH, gt_BA, mask).item()

        # Evaluation of brightest pixel (Spread)
        if not gt_BP==None:
            gt_BP = torch.squeeze(gt_BP, 0)*0.5+0.5
            result['bpMSE_pr_BP_SH'] = util.mse_with_mask(pr_BP_SH, gt_BP, mask).item()

        # Evaluation of brightest coordinate
        gt_BC_list = []
        gt_BC_num = int(gt_BC[0, 0, 3].item())
        for i in range(gt_BC_num):
            gt_BC_list.append((gt_BC[0, i, 0].item(), gt_BC[0, i, 1].item(), int(gt_BC[0, i, 2].item()), int(gt_BC[0, i, 3].item())))
        gt_BC = gt_BC_list

        result['bcDist_pr_BC_SH'] = util.calc_dist(gt_BC, pr_BC_SH)

        result['pr_BC_SH'] = pr_BC_SH[0]

        return result

    def eval_sh(self, mask, gt_SH, pr_SH):
        result = {}

        mask = torch.squeeze(mask, 0)*0.5+0.5
        gt_SH = torch.squeeze(gt_SH, 0)*0.5+0.5
        pr_SH = torch.squeeze(pr_SH, 0)*0.5+0.5
        result['shMSE'] = util.mse_with_mask(pr_SH, gt_SH, mask.expand(gt_SH.size())).item()

        gt_SH_np = gt_SH.to('cpu').detach().numpy().copy()
        pr_SH_np = pr_SH.to('cpu').detach().numpy().copy()

        gt_SH_np = np.transpose(gt_SH_np, (1, 2, 0))
        pr_SH_np = np.transpose(pr_SH_np, (1, 2, 0))

        result['shPSNR'] = compare_psnr(gt_SH_np, pr_SH_np)
        result['shSSIM'] = compare_ssim(gt_SH_np, pr_SH_np, multichannel=True)

        return result

    def label_sh(self):
        label = {}
        label['baMSE'] = ['baMSE_pr_BA_SH']
        label['bpMSE'] = ['bpMSE_pr_BP_SH']
        label['bcDist'] = ['bcDist_pr_BC_SH']
        label['BC'] = ['pr_BC_SH']

        label['shEval'] = ['shMSE', 'shPSNR', 'shSSIM']
        return label

    def eval_bp_pr(self, mask, gt_BA, gt_BP, gt_BC, pr_BA, pr_BP=None, pr_BC=None, suffix=''):
        result = {}

        mask = torch.squeeze(mask, 0)*0.5+0.5
        all_one = torch.ones_like(mask)
        if self.opt.eval_mask_calc_bp:
            mask_bp = mask
        else:
            mask_bp = all_one

        pr_BA = torch.squeeze(pr_BA, 0)*0.5+0.5
        _, _, pr_BP_BA, pr_BC_BA = util.calc_brightest(pr_BA, mask_bp, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(gt_BA, 0)*0.5+0.5
        result['baMSE_pr_BA{}'.format(suffix)] = util.mse_with_mask(pr_BA, gt_BA, mask).item()

        # Evaluation of brightest pixel (Spread)
        if not gt_BP==None:
            gt_BP = torch.squeeze(gt_BP, 0)*0.5+0.5
            result['bpMSE_pr_BP_BA{}'.format(suffix)] = util.mse_with_mask(pr_BP_BA, gt_BP, mask).item()

        # Evaluation of brightest coordinate
        gt_BC_list = []
        gt_BC_num = int(gt_BC[0, 0, 3].item())
        for i in range(gt_BC_num):
            gt_BC_list.append((gt_BC[0, i, 0].item(), gt_BC[0, i, 1].item(), int(gt_BC[0, i, 2].item()), int(gt_BC[0, i, 3].item())))
        gt_BC = gt_BC_list
        pr_BC = [(pr_BC[0, 0].item(), pr_BC[0, 1].item(), 1, 1)]

        result['bcDist_pr_BC_BA{}'.format(suffix)] = util.calc_dist(gt_BC, pr_BC_BA)
        result['bcDist_pr_BC{}'.format(suffix)] = util.calc_dist(gt_BC, pr_BC)

        result['pr_BC_BA{}'.format(suffix)] = pr_BC_BA[0]
        result['pr_BC{}'.format(suffix)] = pr_BC[0]

        if not pr_BP==None:
            pr_BP = torch.squeeze(pr_BP, 0)*0.5+0.5
            _, _, pr_BP_BP, pr_BC_BP = util.calc_brightest(pr_BP, mask_bp, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)
            if not gt_BP==None:
                result['bpMSE_pr_BP_BP{}'.format(suffix)] = util.mse_with_mask(pr_BP_BP, gt_BP, mask).item()
            result['bcDist_pr_BC_BP{}'.format(suffix)] = util.calc_dist(gt_BC, pr_BC_BP)
            result['pr_BC_BP{}'.format(suffix)] = pr_BC_BP[0]

        return result

    def label_pr(self, pr_BP=True, suffix=''):
        label = {}
        label['baMSE'] = ['baMSE_pr_BA{}'.format(suffix)]
        label['bpMSE'] = ['bpMSE_pr_BP_BA{}'.format(suffix), 'bpMSE_pr_BP_BP{}'.format(suffix)]

        if pr_BP:
            label['bcDist'] = ['bcDist_pr_BC_BA{}'.format(suffix), 'bcDist_pr_BC_BP{}'.format(suffix), 'bcDist_pr_BC{}'.format(suffix)]
            label['BC'] = ['pr_BC_BA{}'.format(suffix), 'pr_BC_BP{}'.format(suffix), 'pr_BC{}'.format(suffix)]
        else:
            label['bcDist'] = ['bcDist_pr_BC_BA{}'.format(suffix), 'bcDist_pr_BC{}'.format(suffix)]
            label['BC'] = ['pr_BC_BA{}'.format(suffix), 'pr_BC{}'.format(suffix)]
        return label

    def eval_bp_base(self, mask, gt_BA, gt_BP, gt_BC, input):
        result = {}

        mask = torch.squeeze(mask, 0)*0.5+0.5
        all_one = torch.ones_like(mask)
        if self.opt.eval_mask_calc_bp:
            mask_bp = mask
        else:
            mask_bp = all_one

        input_g = torch.squeeze(torch.mean(input, 1, keepdim=True), 0)*0.5+0.5
        all_half = torch.ones_like(input_g) * 0.5
        all_zero = torch.zeros_like(input_g)
        base_BA_RA, _, base_BP_RA, base_BC_RA = util.calc_brightest(input_g, mask_bp, nr_tap=self.opt.bp_nr_tap, nr_sigma=self.opt.bp_nr_sigma, spread_tap=self.opt.bp_tap, spread_sigma=self.opt.bp_sigma)

        # Evaluation of 20% brightest area
        gt_BA = torch.squeeze(gt_BA, 0)*0.5+0.5
        result['baMSE_base_BA_RA']  = util.mse_with_mask(base_BA_RA, gt_BA, mask).item()
        result['baMSE_base_BA_0'] = util.mse_with_mask(all_zero, gt_BA, mask).item()
        result['baMSE_base_BA_h'] = util.mse_with_mask(all_half, gt_BA, mask).item()
        result['baMSE_base_BA_1'] = util.mse_with_mask(all_one, gt_BA, mask).item()

        # Evaluation of brightest pixel (Spread)
        if not gt_BP==None:
            gt_BP = torch.squeeze(gt_BP, 0)*0.5+0.5
            result['bpMSE_base_BP_RA'] = util.mse_with_mask(base_BP_RA, gt_BP, mask).item()
            result['bpMSE_base_BP_0'] = util.mse_with_mask(all_zero, gt_BP, mask).item()
            result['bpMSE_base_BP_h'] = util.mse_with_mask(all_half, gt_BP, mask).item()
            result['bpMSE_base_BP_1'] = util.mse_with_mask(all_one, gt_BP, mask).item()

        # Evaluation of brightest coordinate
        gt_BC_list = []
        gt_BC_num = int(gt_BC[0, 0, 3].item())
        for i in range(gt_BC_num):
            gt_BC_list.append((gt_BC[0, i, 0].item(), gt_BC[0, i, 1].item(), int(gt_BC[0, i, 2].item()), int(gt_BC[0, i, 3].item())))
        gt_BC = gt_BC_list

        result['bcDist_base_BC_RA'] = util.calc_dist(gt_BC, base_BC_RA)
        result['bcDist_base_BC_05'] = util.calc_dist(gt_BC, [(0.5, 0.5, 1, 1)])

        condition = gt_BC[0][2]
        gt_BC_num = gt_BC[0][3]
        if torch.sum(mask > 0.5) < 1:
            condition = 3
        result['condition'] = condition
        result['gt_BC_num'] = gt_BC_num

        result['gt_BC'] = gt_BC[0]
        result['base_BC_RA'] = base_BC_RA[0]

        return result

    def label_base(self):
        label = {}
        label['baMSE'] = ['baMSE_base_BA_RA', 'baMSE_base_BA_0', 'baMSE_base_BA_h', 'baMSE_base_BA_1']
        label['bpMSE'] = ['bpMSE_base_BP_RA', 'bpMSE_base_BP_0', 'bpMSE_base_BP_h', 'bpMSE_base_BP_1']
        label['bcDist'] = ['bcDist_base_BC_RA', 'bcDist_base_BC_05']
        label['BC'] = ['gt_BC', 'base_BC_RA']
        return label

    def eval_bp(self):
        with torch.no_grad():
            self.forward()     
            self.compute_visuals()

        result = eval_bp()


    def compute_whdr(self, reflectance, judgements, delta=0.1):
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = reflectance.shape[0:2]
 
        error_sum = 0.0
        weight_sum = 0.0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            # convert to grayscale and threshold
            l1 = max(1e-10, np.mean(reflectance[
                int(point1['y'] * rows), int(point1['x'] * cols), ...]))
            l2 = max(1e-10, np.mean(reflectance[
                int(point2['y'] * rows), int(point2['x'] * cols), ...]))

            # # convert algorithm value to the same units as human judgements
            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'


            if darker != alg_darker:
                error_sum += weight

            weight_sum += weight

        if weight_sum:
            return (error_sum / weight_sum)
        else:
            return None

    def evaluate_WHDR(self, prediction_R, targets):
        total_whdr = float(0)
        count = float(0) 

        for i in range(0, prediction_R.size(0)):
            print(targets['path'][i])
            prediction_R_tmp = util.normalize_n1p1_to_0p1(grayscale=False)(prediction_R.data[i,:,:,:])
            prediction_R_np = prediction_R_tmp.cpu().numpy()

            # prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
            prediction_R_np = np.transpose(prediction_R_np, (1,2,0))
            # cv2.imwrite('ref{}.png'.format(i), (prediction_R_np[:,:,::-1]*255.0).astype(np.uint8))

            o_h = targets['oringinal_shape'][0].numpy()
            o_w = targets['oringinal_shape'][1].numpy()
            # resize to original resolution 
            prediction_R_np = resize(prediction_R_np, (o_h[i],o_w[i]), order=1, preserve_range=True)
            # load Json judgement 
            judgements = json.load(open(targets["judgements_path"][i]))
            whdr = self.compute_whdr(prediction_R_np, judgements, 0.1)

            total_whdr += whdr
            count += 1.

        return total_whdr, count

    def evlaute_iiw(self, input_, targets):
        # switch to evaluation mode
        input_ = input_.contiguous().float()
        input_images = input_

        # input_images = Variable(input_.cuda() , requires_grad = False)
        prediction_S, prediction_R , rgb_s = self.netG1.forward(input_images)
        prediction_R = torch.clamp(prediction_R, min=-1, max=1)

        # prediction_R = torch.exp(prediction_R)

        return self.evaluate_WHDR(prediction_R, targets)

    def compute_pr(self, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, thres_count=400):

        thres_list = saw_utils.gen_pr_thres_list(thres_count)
        photo_ids = saw_utils.load_photo_ids_for_split(
            splits_dir=splits_dir, dataset_split=dataset_split)

        AP = []
        # mode = [0, 1]
        mode = [0]
        for m in mode:
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
                photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size, mode=m)

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

            AP.append(np.trapz(plot_arr[:,1], plot_arr[:,0]))

        return AP


    def get_precision_recall_list_new(self, pixel_labels_dir, thres_list, photo_ids,
                                  class_weights, bl_filter_size, mode):

        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)
            for _ in range(output_count)
        ]

        count = 0 
        # eval_num = 1
        eval_num = len(photo_ids)
        total_num_img = eval_num

        # for photo_id in (photo_ids):
        for photo_id in photo_ids[:total_num_img]:
            print("photo_id ", count, photo_id, total_num_img)

            saw_img_ori = saw_utils.load_img_arr(photo_id)
            original_h, original_w = saw_img_ori.shape[0], saw_img_ori.shape[1]
            saw_img = saw_utils.resize_img_arr(saw_img_ori)

            saw_img = np.transpose(saw_img, (2,0,1))
            input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
            input_images = (input_ - 0.5) * 2.0
            # input_images = Variable(input_.cuda() , requires_grad = False)

            # prediction_S, prediction_R, rgb_s = self.netG.forward(input_images) 
            prediction_S, prediction_A, _ = self.netG1(input_images)
            # prediction_A = torch.clamp(prediction_A, min=-1, max=1)
            prediction_S = torch.clamp(prediction_S, min=-1, max=1)
            prediction_S = util.normalize_n1p1_to_0p1(grayscale=True)(prediction_S.data[0,:,:,:])
            # prediction_A = util.normalize_n1p1_to_0p1(grayscale=False)(prediction_A.data[0,:,:,:])
            
            prediction_S_np = prediction_S.data[0,:,:].cpu().numpy()
            prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)
            # prediction_A_np = prediction_A.data[:,:,:].cpu().numpy()
            # prediction_A_np = np.transpose(prediction_A_np, (1, 2, 0))
            # prediction_A_np = resize(prediction_A_np, (original_h, original_w), order=1, preserve_range=True)
            # cv2.imwrite('test_shading.png', prediction_S_np*255)
            # cv2.imwrite('test_albedo.png', cv2.cvtColor((prediction_A_np*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            # cv2.imwrite('test_original.png', cv2.cvtColor((saw_img_ori*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            

            # compute confusion matrix
            conf_mx_list = self.eval_on_images( shading_image_arr = prediction_S_np,
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_id=photo_id, bl_filter_size=bl_filter_size, mode=mode
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


    def eval_on_images(self, shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size, mode):
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

        # Add-------------------------------------

        # diffuclut and harder dataset
        srgb_img = saw_utils.load_img_arr(photo_id)
        srgb_img = np.mean(srgb_img, axis = 2)
        img_gradmag = saw_utils.compute_gradmag(srgb_img)

        smooth_mask = (y_true == 2)
        average_gradient = np.zeros_like(img_gradmag)
        # find every connected component
        labeled_array, num_features = label(smooth_mask)
        for j in range(1, num_features+1):
            # for each connected component, compute the average image graident for the region
            avg = np.mean(img_gradmag[labeled_array == j])
            average_gradient[labeled_array == j]  = avg

        average_gradient = np.ravel(average_gradient)
        # Add-------------------------------------
        
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

            # confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            if mode < 0.1:
                confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            else:
                confusion_matrix = saw_utils.grouped_weighted_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask], average_gradient[~ignored_mask])

            ret.append(confusion_matrix)

        return ret

