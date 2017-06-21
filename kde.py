from __future__ import print_function
import argparse
import random
from scipy.stats import gaussian_kde
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from fileUtil import FileUtil

def convert_to_ndarrays(dataset):
    return np.stack([vec[0].numpy().flatten() for vec in dataset])


def fit_kde(X, bandwidth):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X=X)
    return kde


def cal_logprob(kde, data):
    #logprob_vec =  kde.score_samples(data)
    #mean_logprob = np.mean(logprob_vec)
    #max_p = np.max(logprob_vec)
    #normalized_logp = max_p + np.log(np.mean(np.exp(logprob_vec - max_p))) - (original_data_size - 1) * np.log(sigma * np.sqrt(np.pi * 2))
    return kde.score(data) / data.shape[0]


def search_bandwidth(val_data, cvJobs):
    data = convert_to_ndarrays(val_data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=cvJobs)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    return grid.best_estimator_.bandwidth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--cvJobs', type=int, help='number of jobs for cross validation', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--modelPath', default='', help="path to the kernel density estimation model.")
    parser.add_argument('--task', default='hyper', help="hyper | train")
    parser.add_argument('--normalizeImages', type=bool, default=True)
    opt = parser.parse_args()
    print(opt)


    opt.manualSeed = random.randint(1, 10000) # fix seed
    nc = 3 # number of channels
    if opt.dataset == 'lsun':
        #3x256x341
        if opt.normalizeImages:
            transform_op = transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        else:
            transform_op = transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
            ])
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['tower_train'],
                            transform=transform_op)
        val_dataset = [dataset[i] for i in range(50000, 60000)]
        test_dataset = dset.LSUN(db_path=opt.dataroot, classes=['tower_val'],
                            transform=transform_op)
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        val_dataset = [dataset[i] for i in range(10000, 20000)]
        test_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                                     transform=transforms.Compose([
                                         transforms.Scale(opt.imageSize),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        val_dataset = [dataset[i] for i in range(50000,60000)]

        test_dataset = dset.MNIST(root=opt.dataroot, download=True,train=False,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 1
    assert dataset
    assert val_dataset
    assert test_dataset


    if opt.task == 'hyper':
        search_bandwidth(val_dataset, opt.cvJobs)
    else:
        train_set = convert_to_ndarrays(dataset)
        print('max value: {0} , min value: {1}'.format(np.max(train_set), np.min(train_set)))
        if opt.imageSize == 32:
            b_width = 0.1
        else:
            b_width = 0.12742749857
        kde = fit_kde(train_set, bandwidth=b_width)
        mean_logprob = cal_logprob(kde, convert_to_ndarrays(test_dataset))
        print('mean log probability : {0}'.format(mean_logprob))
    # MNIST, size 64, bandwidth 0.206913808111
    # MNIST size 32 unnormalized 0.1 logprob 880.783584576
    # MNIST, size 28, bandwidth 0.263665089873 unnormalized 0.12742749857 logprob 526.829087276
    # CIFAR10, 32, bandwidth 0.263665089873

if __name__ == '__main__':
    main()