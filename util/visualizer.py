import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import cv2
import torch
from collections import OrderedDict


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def calc_brightest_portions(visuals, name, disp, opt):
    img = torch.squeeze(visuals[name], 0)

    img_gray = torch.mean(img, 0, keepdim=True)
    img_gray = util.normalize_n1p1_to_0p1(grayscale=True)(img_gray)
    mask = torch.ones_like(img_gray)
    brightest_area, _, brightest_pixel, _ = util.calc_brightest(img_gray, mask, nr_tap=opt.bp_nr_tap, nr_sigma=opt.bp_nr_sigma, spread_tap=opt.bp_tap, spread_sigma=opt.bp_sigma)

    brightest_area = util.normalize_0p1_to_n1p1(grayscale=True)(brightest_area)
    brightest_area = torch.unsqueeze(brightest_area, 0)
    brightest_pixel = util.normalize_0p1_to_n1p1(grayscale=True)(brightest_pixel)
    brightest_pixel = torch.unsqueeze(brightest_pixel, 0)
    visuals['pr_BA_{}'.format(disp)] = brightest_area
    visuals['pr_BP_{}'.format(disp)] = brightest_pixel

    return visuals

def mask_on_image(src, visuals):
    mask = util.tensor2im(visuals['mask'])
    src = src.astype(np.float32) * mask.astype(np.float32) / 255.0
    return src.astype(np.uint8)


def jet_on_image(src, visuals, mode='alpha', alpha=0.8):
    # jet = cv2.applyColorMap(src, cv2.COLORMAP_JET)
    jet = cv2.applyColorMap(src, cv2.COLORMAP_TURBO)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)                
    image = util.tensor2im(visuals['input'])
    
    if mode=='alpha':
        alpha = alpha
        out = cv2.addWeighted(jet, alpha, image, 1 - alpha, 0)
    else:
        thr = 25
        mask = np.zeros_like(src)
        mask[src>thr] = 1
        invmask = 1 - mask
        out = jet*mask + image*invmask
    return out

def postprocess(img, visuals, label, resize=False):
    # mask_label = ['pr_BA', 'pr_BA2', 'pr_BP', 'pr_BP2']
    jet_label = ['gt_BA', 'pr_BA_RA', 'pr_BA_SH', 'pr_BA', 'pr_BA2', 'pr_BP', 'pr_BP2']
    point_label = ['gt_BP', 'pr_BP_RA', 'pr_BP_SH',  'pr_BP_BP', 'pr_BP_BP2', 'pr_BP_BC', 'pr_BP_BC2', 'gt_BC', 'pr_BC']

    # if label in mask_label:
        # img = mask_on_image(img, visuals)
    if label in jet_label:
        img = jet_on_image(img, visuals)
    if label in point_label:
        img = jet_on_image(img, visuals, alpha=0.5)
        
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if resize:
        img = cv2.resize(img, (320, 240))
    
    return img


def save_images(webpage, visuals, image_path, opt, aspect_ratio=1.0, width=256, gain=1.0, multi=True, multi_ch=25, resize=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    if opt.disp_brighest_info:
        visuals = calc_brightest_portions(visuals, 'input', 'RA',  opt) # GT Radiance
        visuals = calc_brightest_portions(visuals, 'pr_SH', 'SH',  opt) # GT Radiance

    if multi == True:
        for c in range(multi_ch):
            ims, txts, links = [], [], []
            for label, im_data in visuals.items():
                im = util.tensor2im(im_data, gain=gain, ch=c)
                im = postprocess(im, visuals, label, resize)
                image_name = '%s_%s_%s.png' % (name, label, str(c).zfill(2))
                save_path = os.path.join(image_dir, image_name)
                util.save_image(im, save_path, aspect_ratio=aspect_ratio)
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
            webpage.add_images(ims, txts, links, width=width)
    else:
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data, gain=gain)
            im = postprocess(im, visuals, label, resize)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=width)



class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, resize=False):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.opt.disp_brighest_info:
            visuals = calc_brightest_portions(visuals, 'input', 'RA',  self.opt) # GT Radiance
            visuals = calc_brightest_portions(visuals, 'pr_SH', 'SH',  self.opt) # GT Radiance
            
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    image_numpy = postprocess(image_numpy, visuals, label, resize)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        image_numpy = postprocess(image_numpy, visuals, label, resize)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            print('Updated images...')
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                image_numpy = postprocess(image_numpy, visuals, label, resize)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=60)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    # image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
