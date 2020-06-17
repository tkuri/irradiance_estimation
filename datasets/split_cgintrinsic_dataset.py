import argparse
import pickle
import random

def main():
    random.seed(101)
    dataroot = '//jpc00160593/Users/kurita/dataset/CGIntrinsics'
    list_dir = dataroot + '/intrinsics_final/train_list/'
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )
    num_images = len(images_list)
    test_num = num_images//10 #Train:Test=9:1
    test_list = random.sample(images_list, test_num)
    train_list = list(set(images_list)^set(test_list))

    print('images_list len:', len(images_list))
    print('train_list len:', len(train_list))
    print(train_list[:10])
    print('test_list len:', len(test_list))
    print(test_list[:10])

    with open(list_dir + "img_batch_train.p", mode='wb') as f:
        pickle.dump(train_list, f)

    with open(list_dir + "img_batch_test.p", mode='wb') as f:
        pickle.dump(test_list, f)

if __name__ == '__main__':
    main()