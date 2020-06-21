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

    # ba_mse_ra = df['ba_mse_ra'].values
    # ba_mse_sh = df['ba_mse_sh'].values
    # ba_mse_ba = df['ba_mse_ba'].values
    # result += [['ba_mse_ra', '{:.5f}'.format(np.mean(ba_mse_ra)), '{:.5f}'.format(np.median(ba_mse_ra))]]
    # result += [['ba_mse_sh', '{:.5f}'.format(np.mean(ba_mse_sh)), '{:.5f}'.format(np.median(ba_mse_sh))]]
    # result += [['ba_mse_ba', '{:.5f}'.format(np.mean(ba_mse_ba)), '{:.5f}'.format(np.median(ba_mse_ba))]]

    # bp_mse_ra = df['bp_mse_ra'].values
    # bp_mse_sh = df['bp_mse_sh'].values
    # bp_mse_ba = df['bp_mse_ba'].values
    # bp_mse_bp = df['bp_mse_bp'].values
    # result += [['BP MSE Evaluation', 'mean', 'meadian']]
    # result += [['bp_mse_ra', '{:.5f}'.format(np.mean(bp_mse_ra)), '{:.5f}'.format(np.median(bp_mse_ra))]]
    # result += [['bp_mse_sh', '{:.5f}'.format(np.mean(bp_mse_sh)), '{:.5f}'.format(np.median(bp_mse_sh))]]
    # result += [['bp_mse_ba', '{:.5f}'.format(np.mean(bp_mse_ba)), '{:.5f}'.format(np.median(bp_mse_ba))]]
    # result += [['bp_mse_bp', '{:.5f}'.format(np.mean(bp_mse_bp)), '{:.5f}'.format(np.median(bp_mse_bp))]]

    df = df[df['condition'] == 1]
    result += [['Modified data length for coordinate evaluation.:', len(df)]]
    result += [['BC Coodinate Distance Evaluation', 'mean', 'median', '<10%', '<20%', '<30%']]
    for label in dist_labels:
        values = df[label].values
        num = len(values)
        result += [[label, '{:.3f}'.format(np.mean(values)), '{:.3f}'.format(np.median(values)), '{:.3f}'.format(np.sum(values<0.1)/num), '{:.3f}'.format(np.sum(values<0.2)/num), '{:.3f}'.format(np.sum(values<0.3)/num)]]

    # dist_ra = df['dist_ra'].values
    # dist_sh = df['dist_sh'].values
    # dist_ba = df['dist_ba'].values
    # dist_bp = df['dist_bp'].values
    # dist_bc = df['dist_bc'].values
    # dist_05 = df['dist_05'].values

    # num = len(dist_ra)

    # result += [['dist_ra', '{:.3f}'.format(np.mean(dist_ra)), '{:.3f}'.format(np.median(dist_ra)), '{:.3f}'.format(np.sum(dist_ra<0.1)/num), '{:.3f}'.format(np.sum(dist_ra<0.2)/num), '{:.3f}'.format(np.sum(dist_ra<0.3)/num)]]
    # result += [['dist_sh', '{:.3f}'.format(np.mean(dist_sh)), '{:.3f}'.format(np.median(dist_sh)), '{:.3f}'.format(np.sum(dist_sh<0.1)/num), '{:.3f}'.format(np.sum(dist_sh<0.2)/num), '{:.3f}'.format(np.sum(dist_sh<0.3)/num)]]
    # result += [['dist_ba', '{:.3f}'.format(np.mean(dist_ba)), '{:.3f}'.format(np.median(dist_ba)), '{:.3f}'.format(np.sum(dist_ba<0.1)/num), '{:.3f}'.format(np.sum(dist_ba<0.2)/num), '{:.3f}'.format(np.sum(dist_ba<0.3)/num)]]
    # result += [['dist_bp', '{:.3f}'.format(np.mean(dist_bp)), '{:.3f}'.format(np.median(dist_bp)), '{:.3f}'.format(np.sum(dist_bp<0.1)/num), '{:.3f}'.format(np.sum(dist_bp<0.2)/num), '{:.3f}'.format(np.sum(dist_bp<0.3)/num)]]
    # result += [['dist_bc', '{:.3f}'.format(np.mean(dist_bc)), '{:.3f}'.format(np.median(dist_bc)), '{:.3f}'.format(np.sum(dist_bc<0.1)/num), '{:.3f}'.format(np.sum(dist_bc<0.2)/num), '{:.3f}'.format(np.sum(dist_bc<0.3)/num)]]
    # result += [['dist_05', '{:.3f}'.format(np.mean(dist_05)), '{:.3f}'.format(np.median(dist_05)), '{:.3f}'.format(np.sum(dist_05<0.1)/num), '{:.3f}'.format(np.sum(dist_05<0.2)/num), '{:.3f}'.format(np.sum(dist_05<0.3)/num)]]

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

    # n_ra, bins_ra, _ = ax[0].hist(dist_ra, bins=20, range=(0, 1.0), alpha=0.7, label='dist ra')
    # n_sh, bins_sh, _ = ax[1].hist(dist_sh, bins=20, range=(0, 1.0), alpha=0.7, label='dist sh')
    # n_ba, bins_ba, _ = ax[2].hist(dist_ba, bins=20, range=(0, 1.0), alpha=0.7, label='dist ba')
    # n_bp, bins_bp, _ = ax[3].hist(dist_bp, bins=20, range=(0, 1.0), alpha=0.7, label='dist bp')
    # n_bc, bins_bc, _ = ax[4].hist(dist_bc, bins=20, range=(0, 1.0), alpha=0.7, label='dist bc')
    # n_05, bins_05, _ = ax[5].hist(dist_05, bins=20, range=(0, 1.0), alpha=0.7, label='dist bc')
    # for i in range(dist_num):
    #     ax[i].set_ylim(0, 500)
    #     ax[i].legend()

    # y2_ra = np.add.accumulate(n_ra) / n_ra.sum()
    # y2_sh = np.add.accumulate(n_sh) / n_sh.sum()
    # y2_ba = np.add.accumulate(n_ba) / n_ba.sum()
    # y2_bp = np.add.accumulate(n_bp) / n_bp.sum()
    # y2_bc = np.add.accumulate(n_bc) / n_bc.sum()
    # y2_05 = np.add.accumulate(n_05) / n_05.sum()
    # x2 = np.convolve(bins_ra, np.ones(2) / 2, mode="same")[1:]

    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # lines = ax.plot(x2, y2_ra, ls='--', marker='o',
    #         label='dist ra')
    # lines = ax.plot(x2, y2_sh, ls='--', marker='o',
    #         label='dist sh')
    # lines = ax.plot(x2, y2_ba, ls='--', marker='*',
    #         label='dist ba')
    # lines = ax.plot(x2, y2_bp, ls='--', marker='*',
    #         label='dist bp')
    # lines = ax.plot(x2, y2_bc, ls='--', marker='*',
    #         label='dist bc')
    # lines = ax.plot(x2, y2_05, ls='--', marker='o',
    #         label='dist 05')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(visible=False)

    # plt.show()
    plt.title(os.path.basename(args.result_name))
    plt.savefig('{}.png'.format(args.result_name))

if __name__ == '__main__':
    main()