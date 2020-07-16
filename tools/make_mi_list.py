
import glob
import os
import pickle
import random
import argparse

root = '//JPC00160593/Users/kurita/dataset/Multi-Illumination/'

def parser():
    parser = argparse.ArgumentParser(description='Make multi-illination dataset.')
    parser.add_argument('listname', help='Output list name')
    parser.add_argument('--num', type=int, default=4, help='Sampling num')
    args = parser.parse_args()
    return args


def main():
    args = parser()
    list_dir = root + 'train_list/'
    data_dir = root + 'data/'

    dirs = glob.glob(data_dir+'*')
    dirs = [os.path.basename(dir) for dir in dirs]

    pick = random.sample(dirs, args.num)
    print('Pickuped {} samples'.format(args.num))

    with open(list_dir + args.listname, 'wb') as f:
        pickle.dump(pick, f)

if __name__ == "__main__":
    main()
