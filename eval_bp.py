import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv, os, argparse


def parser():
    parser = argparse.ArgumentParser(description='Extract max region from irradiance image')
    parser.add_argument('fname', help='Input file name')
    parser.add_argument('result_name', help='Result file name')
    args = parser.parse_args()
    return args

def main():
    args = parser()
    # df = pd.read_csv('./results/cgintrinsic/brightest_resnet/test_latest/bp_eval_epoch3.csv')
    df = pd.read_csv(args.fname)
    labels = df.columns
    ba_mse_labels = [s for s in labels if 'ba_mse' in s]
    bp_mse_labels = [s for s in labels if 'bp_mse' in s]
    dist_labels = [s for s in labels if 'dist' in s]
    print(labels)
    result = [['Original data length:', len(df)]]
    df = df[df['condition'] != 0]
    result += [['Modified data length for brightest area/pixel evaluation:', len(df)]]
    result += [['BA MSE Evaluation', 'mean', 'meadian']]
    for label in ba_mse_labels:
        values = df[label].values
        result += [[label, '{:.5f}'.format(np.mean(values)), '{:.5f}'.format(np.median(values))]]
    result += [['BP MSE Evaluation', 'mean', 'meadian']]
    for label in bp_mse_labels:
        values = df[label].values
        result += [[label, '{:.5f}'.format(np.mean(values)), '{:.5f}'.format(np.median(values))]]

    # df = df[df['condition'] != 0]
    # df = df[df['condition'] == 1]
    # result += [['Modified data length for coordinate evaluation.:', len(df)]]
    result += [['BC Coodinate Distance Evaluation', 'mean', 'median', '<10%', '<20%', '<30%']]
    for label in dist_labels:
        values = df[label].values
        num = len(values)
        result += [[label, '{:.3f}'.format(np.mean(values)), '{:.3f}'.format(np.median(values)), '{:.3f}'.format(np.sum(values<0.1)/num), '{:.3f}'.format(np.sum(values<0.2)/num), '{:.3f}'.format(np.sum(values<0.3)/num)]]

    with open('{}.csv'.format(args.result_name), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)


    sns.set(style="darkgrid", palette="muted", color_codes=True)
    dist_num = len(dist_labels)

    fig, ax = plt.subplots(1, dist_num, figsize=(dist_num*6, 5))
    y2 = []

    for i, label in enumerate(dist_labels):
        values = df[label].values
        n, bins, _ = ax[i].hist(values, bins=20, range=(0, 1.0), alpha=0.7, label=label)
        y2.append(np.add.accumulate(n) / n.sum())
        if i==0:
            x2 = np.convolve(bins, np.ones(2) / 2, mode="same")[1:]
        ax[i].set_ylim(0, 500)
        ax[i].legend()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    for i, label in enumerate(dist_labels):
        lines = ax.plot(x2, y2[i], ls='--', marker='o',
                label=label)

    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(visible=False)

    # plt.show()
    plt.title(os.path.basename(args.result_name))
    plt.savefig('{}.png'.format(args.result_name))

if __name__ == '__main__':
    main()