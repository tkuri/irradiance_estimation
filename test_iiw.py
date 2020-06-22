import time
import torch
import numpy as np
from options.test_options import TestOptions
import sys, traceback
from models import create_model
from data.data_loader import CreateDataLoaderIIWTest

def test_iiw(model):
    total_loss =0.0
    total_loss_eq =0.0
    total_loss_ineq =0.0

    total_count = 0.0
    # model.switch_to_eval()
    model.eval()

    for j in range(0,3):
        print("============================= Validation IIW TESTSET ============================", j)

        data_loader_IIW_TEST = CreateDataLoaderIIWTest(full_root, test_list_dir, j)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for i, data in enumerate(dataset_iiw_test):
            stacked_img = data['img_1']
            targets = data['target_1']

            total_whdr, count = model.evlaute_iiw(stacked_img, targets)
            total_loss  += total_whdr
            total_count += count
            print('i, whdr, count', i, total_whdr, count)

    print("IIW TEST WHDR %f"%(total_loss/(total_count)))

if __name__ == '__main__':
    # root = "/"    
    # full_root = root +'/phoenix/S6/zl548//IIW/iiw-dataset/data/'
    root = ""
    full_root = root +'//JPC00160593/Users/kurita/dataset/CGIntrinsics/'
    # full_root = root +'//JPC00160593/Users/kurita/dataset/CGIntrinsics/IIW/iiw-dataset/data/'
    test_list_dir = '//JPC00160593/Users/kurita/dataset/CGIntrinsics/IIW/test_list/'

    opt = TestOptions().parse()
    model = create_model(opt)

    print("WE ARE IN TESTING IIW!!!!")
    test_iiw(model)
