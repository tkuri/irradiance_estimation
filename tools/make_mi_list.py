
import glob
import os
import pickle
import random
import argparse

root = '//JPC00160593/Users/kurita/dataset/Multi-Illumination/'

def parser():
    parser = argparse.ArgumentParser(description='Make multi-illination dataset.')
    parser.add_argument('listname', help='Output list name')
    parser.add_argument('--train_num', type=int, default=4, help='Sampling train num')
    parser.add_argument('--test_num', type=int, default=2, help='Sampling test num')
    parser.add_argument('--mode', type=str, default='train', help='[train/train_test]')
    args = parser.parse_args()
    return args

def make_train(list_dir, dirs):
    pick = random.sample(dirs, args.train_num)
    print('Pickuped {} samples'.format(args.train_num))

    with open(list_dir + args.listname, 'wb') as f:
        pickle.dump(pick, f)

def make_train_test(list_dir, dirs):
    num = args.train_num + args.test_num
    pick = random.sample(dirs, num)
    print('Pickuped {} samples'.format(num))
    pick_train = pick[:args.train_num]
    pick_test = pick[args.train_num:]

    with open(list_dir + args.listname + '_train.txt', 'wb') as f:
        pickle.dump(pick_train, f)

    with open(list_dir + args.listname + '_test.txt', 'wb') as f:
        pickle.dump(pick_test, f)

def main():
    list_dir = root + 'train_list/'
    data_dir = root + 'data/'

    dirs = glob.glob(data_dir+'*')
    dirs = [os.path.basename(dir) for dir in dirs]

    if args.mode=='train':
        make_train(list_dir, dirs)
    else:
        make_train_test(list_dir, dirs)



if __name__ == "__main__":
    args = parser()
    main()
