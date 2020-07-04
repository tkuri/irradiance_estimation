import numpy as np
import csv, os, argparse
import cv2

def parser():
    parser = argparse.ArgumentParser(description='Extract max region from irradiance image')
    parser.add_argument('csv_name', help='Input file name')
    parser.add_argument('img_name', help='Input file name')
    parser.add_argument('result_name', help='Result file name')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha blend value')
    args = parser.parse_args()
    return args

def main():
    args = parser()
    lux = np.loadtxt(args.csv_name, delimiter=',')
    img = cv2.imread(args.img_name)
    lux = lux / lux.max() * 255.0
    lux = lux.astype('uint8')

    jet = cv2.applyColorMap(lux, cv2.COLORMAP_TURBO)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    print(jet.shape)
    print(img.shape)
    alpha = args.alpha
    out = cv2.addWeighted(jet, alpha, img, 1 - alpha, 0)
    cv2.imwrite(args.result_name, out)


if __name__ == '__main__':
    main()