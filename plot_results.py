import matplotlib.pyplot as plt

import csv
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import argparse

def read(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        log_probs = []
        for row in reader:
            log_probs.append(float(row[1]))

        assert len(log_probs) == 11, 'length %s in %s ' % (len(log_probs), file_path)
        return log_probs

def read_files(file_dir):
    file_paths = [join(file_dir, f) for f in listdir(file_dir) if isfile(join(file_dir, f))]
    log_prob_matrix = [read(file) for file in file_paths]
    std_dev = np.std(log_prob_matrix, axis=0)
    mean = np.mean(log_prob_matrix, axis=0)
    print(file_dir)
    print('mean: {0}'.format(mean))
    print('std: {0}'.format(std_dev))
    return mean, std_dev

def plot_results(main_dir, target_dir):
    mean_std_tuples = [read_files(join(main_dir, folder)) for folder in listdir(main_dir) if isdir(join(main_dir, folder))]
    mean, std_dev = zip(*mean_std_tuples)
    mus = range(0,11)
    plt.gca().set_color_cycle(['black', 'orange', 'deeppink', 'green', 'blue'])
    fmts = ['>-', 'o-', 'x-', 'D-', 'p-']
    exps = [folder for folder in listdir(main_dir) if isdir(join(main_dir, folder))]
    for i in range(len(mean)):
        plt.errorbar(mus, mean[i], yerr=std_dev[i], fmt=fmts[i])

    plt.legend(exps, loc='lower right')
    plt.xlim((-0.2,10.2))
    #plt.axis((0,11, 0.1, 0.8))
    plt.grid(True)

    x_ticks = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1']
    plt.xticks(mus, x_ticks)

    plt.xlabel('mu')
    plt.ylabel('Log Probability')
    #plt.show()
    plt.savefig(join(target_dir, "mu_relu.png"), bbox_inches='tight')

def barplot(target_dir):
    softplus_gan_dcgan = [523.749325889, 612.327717494, 582.548003606]
    softplus_wgan_dcgan = [511.132247894, 527.427932984, 442.807643129]
    softplus_gan_mlp = [120.218675105, 118.340705537, 144.120714021]
    softplus_wgan_mlp = [453.721877243, 501.204041358, 444.071567208]
    leastSquare_gan_dcgan = [632.135430781, 662.210057543, 669.320158872]
    leastSquare_wgan_dcgan = [528.636335214, 560.550922099, 544.230405069]
    leastSquare_gan_mlp = [231.680103473, 163.158618896, 197.583367488]
    leastSquare_wgan_mlp = [423.829548358, 507.111113457, 484.804082098]

    N=4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    softplus_mean = [np.mean(softplus_gan_dcgan), np.mean(softplus_wgan_dcgan), np.mean(softplus_gan_mlp), np.mean(softplus_wgan_mlp)]
    softplus_std = [np.std(softplus_gan_dcgan), np.std(softplus_wgan_dcgan), np.std(softplus_gan_mlp), np.std(softplus_wgan_mlp)]
    rects_softplus = ax.bar(ind, softplus_mean, width, color='lightblue', yerr=softplus_std)

    ls_mean = [np.mean(leastSquare_gan_dcgan), np.mean(leastSquare_wgan_dcgan), np.mean(leastSquare_gan_mlp),
                     np.mean(leastSquare_wgan_mlp)]
    ls_std = [np.std(leastSquare_gan_dcgan), np.std(leastSquare_wgan_dcgan), np.std(leastSquare_gan_mlp),
                    np.std(leastSquare_wgan_mlp)]
    rects_ls = ax.bar(ind + width, ls_mean, width, color='lightgreen', yerr=ls_std)

    relu_mean = [681.01717885, 542.51281229, 457.24837393, 413.81168954]
    relu_std = [25.93946989, 21.66889571, 35.52057633, 125.71019851]
    rects_relu = ax.bar(ind + 2*width, relu_mean, width, color='yellow', yerr=relu_std)


    ax.set_ylabel('Log Probability')
    ax.set_xticks(ind + 1.5* width)
    ax.set_xticklabels(('gan_dcgan', 'wgan_dcgan', 'gan_mlp', 'wgan_mlp'))

    ax.legend((rects_softplus[0], rects_ls[0],rects_relu[0]), ('Softplus', 'LeastSquare', 'ReLU'))

    # plt.show()
    plt.savefig(join(target_dir, "sp_ls.png"), bbox_inches='tight')


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/3., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects_softplus)
    autolabel(rects_ls)
    autolabel(rects_relu)

def discriminator_barplot(target_dir):
    matsu_gan_dcgan = [676.460141029,705.85256067,700.876242004]
    matsu_wgan_dcgan = [564.063737896,524.636358032,543.634910961]
    matsu_gan_mlp = [461.349531434,519.603921001,513.949371208]
    matsu_wgan_mlp = [360.227033106,392.627267256,361.221227421]

    N=4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    matsu_mean = [np.mean(matsu_gan_dcgan), np.mean(matsu_wgan_dcgan), np.mean(matsu_gan_mlp), np.mean(matsu_wgan_mlp)]
    matsu_std = [np.std(matsu_gan_dcgan), np.std(matsu_wgan_dcgan), np.std(matsu_gan_mlp), np.std(matsu_wgan_mlp)]
    rects_matsu = ax.bar(ind, matsu_mean, width, color='lightblue', yerr=matsu_std)

    relu_mean = [681.01717885, 542.51281229, 457.24837393, 413.81168954]
    relu_std = [25.93946989, 21.66889571, 35.52057633, 125.71019851]
    rects_relu = ax.bar(ind + width, relu_mean, width, color='lightgreen', yerr=relu_std)


    ax.set_ylabel('Log Probability')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('gan_dcgan', 'wgan_dcgan', 'gan_mlp', 'wgan_mlp'))

    ax.legend((rects_matsu[0], rects_relu[0]), ('Matsushita', 'Standard'))

    # plt.show()
    plt.savefig(join(target_dir, "discriminator_last_layer.png"), bbox_inches='tight')


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects_matsu)
    autolabel(rects_relu)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True, help='path to results')
    parser.add_argument('--target_dir', required=True, help='path to target folder')
    opt = parser.parse_args()
    plot_results(opt.result_dir, opt.target_dir)
    barplot(opt.target_dir)
    discriminator_barplot(opt.target_dir)

if __name__ == '__main__':
    main()