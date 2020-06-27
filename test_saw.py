import time
import torch
import numpy as np
from options.test_options import TestOptions
import sys, traceback
from models import create_model


def test_SAW(model):
    # parameters for SAW 
    pixel_labels_dir = saw_root + 'saw/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = saw_root + 'saw/saw_splits'
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    print("============================= Validation ON SAW============================")
    # model.switch_to_eval()
    model.eval()
    AP = model.compute_pr(pixel_labels_dir, splits_dir,
                dataset_split, class_weights, bl_filter_size)

    # print("SAW test AP: {:.4f}, {:.4f}".format(AP[0], AP[1]))
    print("SAW test AP: {:.4f}".format(AP[0]))
    return AP

if __name__ == '__main__':
    # opt = TrainOptions().parse()
    opt = TestOptions().parse()
    root = ''
    # saw_root = root + "/phoenix/S6/zl548/SAW/saw_release/"
    saw_root = root + "//JPC00160593/Users/kurita/dataset/CGIntrinsics/SAW/saw_release/"

    dataset_split = 'E' # Test set

    model = create_model(opt)
    model.setup(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    best_epoch = 0

    print("WE ARE IN TESTING SAW")
    AP = test_SAW(model)
    # with open(opt.result_name + '_saw_{:.4f}_{:.4f}.txt'.format(AP[0], AP[1]), mode='w') as f:
    with open(opt.result_name + '_saw_{:.4f}.txt'.format(AP[0]), mode='w') as f:
        f.write(str(AP))