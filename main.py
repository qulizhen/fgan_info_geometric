from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import models.dcgan as dcgan
import models.mlp as mlp
from models import gan
from fileUtil import FileUtil
from kde import cal_logprob
from kde import fit_kde
from kde import convert_to_ndarrays
import numpy as np
import math
import time
import logging
import csv
import traceback


def train(opt, log_file_path):
    if opt.experiment is None:
        opt.experiment = 'samples'
        os.system('mkdir {0}'.format(opt.experiment))
    elif not os.path.exists(opt.experiment):
        os.system('mkdir {0}'.format(opt.experiment))

    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(opt)

    #opt.manualSeed = random.randint(1, 10000)  # fix seed
    logger.info("Random Seed: %s " % opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    nc = int(opt.nc)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    def eval_with_KDE(generator, original_test_set, random_test_noise):
        num_insts = random_test_noise.size()[0]
        num_batches = int(math.ceil(num_insts / float(opt.batchSize)))
        instances = []
        for i in range(num_batches):
            fake_test_set = generator(
                Variable(random_test_noise[i * opt.batchSize: (i + 1) * opt.batchSize], volatile=True))
            instances.extend([vec.flatten() for vec in fake_test_set.data.cpu().numpy()])
        flattened_data = np.stack(instances)
        # print('input data for KDE is of shape {0} '.format(flattened_data.shape))
        kde = fit_kde(flattened_data, bandwidth=opt.bandwidth)
        mean_logp = cal_logprob(kde, original_test_set)
        return mean_logp

    def xavier_init(param):
        size = param.data.size()
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        param.data = torch.randn(*size) * xavier_stddev

    sample_validation_without_replacement = True
    val_set = []
    if opt.dataset == 'lsun':
        opt.normalizeImages = True
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
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['{0}_train'.format(opt.subset)],
                            transform=transform_op)

        if opt.task == 'hyper':
            dataset = [dataset[i] for i in range(10000, 40000)]

        test_dataset = dset.LSUN(db_path=opt.dataroot, classes=['{0}_val'.format(opt.subset)],
                            transform=transform_op)
        val_set = convert_to_ndarrays([dataset[i] for i in range(0, 1000)])
        if sample_validation_without_replacement:
            dataset = [dataset[i] for i in range(1001, len(dataset))]
        nc = 3
        opt.bandwidth = 0.335981828628
        size_test_noise = 3000
        size_val_noise = 3000
    elif opt.dataset == 'cifar10':
        opt.normalizeImages = True
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        test_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                                    transform=transforms.Compose([
                                        transforms.Scale(opt.imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        print('number of images in test set %s ' % len(test_dataset))
        val_set = convert_to_ndarrays([test_dataset[i] for i in range(0, 1000)])
        test_dataset = [test_dataset[i] for i in range(1001, len(test_dataset))]
        if sample_validation_without_replacement:
            dataset = [dataset[i] for i in range(1001, len(dataset))]
        nc = 3
        size_test_noise = 6000
        size_val_noise = 6000
        if opt.imageSize == 32:
            opt.bandwidth = 0.263665089873
        else:
            opt.bandwidth = 0.335981828628
    elif opt.dataset == 'mnist':

        if opt.normalizeImages:
            img_transform = transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            img_transform = transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.ToTensor(),
            ])

        dataset = dset.MNIST(root=opt.dataroot, download=True,
                             transform=img_transform)
        test_dataset = dset.MNIST(root=opt.dataroot, download=True, train=False,
                                  transform=img_transform)
        nc = 1

        if opt.imageSize == 32:
            opt.bandwidth = 0.1
        else:
            opt.bandwidth = 0.12742749857
        val_set = convert_to_ndarrays([dataset[i] for i in range(50000, 51000)])
        if sample_validation_without_replacement:
            dataset = [dataset[i] for i in range(0, len(dataset)) if i < 50000 or i > 51000]
        size_test_noise = 16000
        size_val_noise = 5000

        # if opt.A in ['mlp']:
        #     logger.info('Apply experimental setting of F-GAN on MNIST.')
        #     opt.nz = 100
        #     opt.ndf = 240
        #     opt.ngf = 1200
        #     opt.init_z = 'uniform_one'
        #     opt.adam = True
        #     opt.lrD = 0.0002
        #     opt.lrG = 0.0002
        #     opt.beta1 = 0.5
        #     opt.batchSize = 4096
        #     opt.init_w = 'uniform'
        #     opt.last_layer = 'sigmoid'
        #     opt.clamp_lower = 0
        #     opt.clamp_upper = 0

    assert dataset
    assert test_dataset
    assert len(val_set) > 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = 1
    if opt.gpu_id < 0:
        ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    n_extra_layers = int(opt.n_extra_layers)

    # Load KDE model, if available

    def init_z(tensor):
        if opt.init_z == 'uniform_one':
            tensor.uniform_(-1, 1)
        elif opt.init_z == 'uniform_zero_one':
            tensor.uniform_(0, 1)
        else:
            tensor.normal_(0, 1)
        return tensor

    test_noise = None
    val_noise = None


    if opt.bandwidth != 0:
        test_noise = init_z(torch.FloatTensor(size_test_noise, nz, 1, 1))
        val_noise = init_z(torch.FloatTensor(size_val_noise, nz, 1, 1))
        test_set = convert_to_ndarrays(test_dataset)
        if opt.cuda:
            test_noise = test_noise.cuda()
            val_noise = val_noise.cuda()

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # initialize generator

    if opt.A == 'wmlp':
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu, hidden_activation=opt.H, mu=opt.mu, last_layer=opt.last_layer)
        if opt.init_w == 'xavier':
            [xavier_init(param) for param in netG.parameters()]
        else:
            [param.data.uniform_(-0.05, 0.05) for param in netG.parameters()]
    elif opt.A == 'mlp':
        netG = gan.GAN_G(opt.imageSize, nz, nc, ngf, ngpu, hidden_activation=opt.H, mu=opt.mu, last_layer=opt.last_layer)
        if opt.init_w == 'xavier':
            [xavier_init(param) for param in netG.parameters()]
        else:
            [param.data.uniform_(-0.05, 0.05) for param in netG.parameters()]
    else:
        if opt.noBN:
            netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        else:
            last_layer = 'sigmoid'
            if opt.normalizeImages:
                last_layer = 'tanh'
            netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers, hidden_activation=opt.H, mu=opt.mu, last_layer=last_layer)
        netG.apply(weights_init)

    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    # Initialize critic
    if opt.A == 'wmlp':
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
        if opt.init_w == 'xavier':
            [xavier_init(param) for param in netG.parameters()]
        else:
            [param.data.uniform_(-0.005, 0.005) for param in netG.parameters()]
    elif opt.A == 'mlp':
        netD = gan.GAN_D(opt.imageSize, nz, nc, ngf, ngpu,hidden_activation = opt.c_activation, last_layer=opt.critic_last_layer, alpha=opt.alpha)
        if opt.init_w == 'xavier':
            [xavier_init(param) for param in netG.parameters()]
        else:
            [param.data.uniform_(-0.005, 0.005) for param in netG.parameters()]
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers, last_layer=opt.critic_last_layer)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    num_images = 24
    fixed_noise = init_z(torch.FloatTensor(num_images, nz, 1, 1))
    one = torch.FloatTensor([1])
    mone = one * -1

    # for GAN
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss()

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        criterion.cuda()
        label = label.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        logger.info("Use ADAM")
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.weightDecay)
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.weightDecay)
    else:
        logger.info("Use RMSprop")
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    def sample_image_compute_density(start_epoch, end_epoch):

        with open(os.path.join(opt.kde_result_dir, 'kde_results.csv'), 'w') as kde_file:
            best_logprob = -10000000
            for epoch in range(start_epoch, end_epoch):
                netG_model = '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch)
                if os.path.exists(netG_model):
                    netG.load_state_dict(torch.load(netG_model))
                    if opt.cuda:
                        netG.cuda()
                    fake = netG(Variable(fixed_noise, volatile=True))
                    if opt.normalizeImages:
                        fake.data = fake.data.mul(0.5).add(0.5)
                    vutils.save_image(fake.data,
                                      '{0}/fake_samples_epoch_{1}.png'.format(opt.kde_result_dir, epoch))
                    logprob_mean = eval_with_KDE(netG, test_set, test_noise)
                    kde_file.write("{0}\t{1}".format(epoch, logprob_mean))
                    best_logprob = max(best_logprob, logprob_mean)
            return best_logprob

    best_logprob = 0
    label = Variable(label)
    noisev = Variable(noise)
    def eval_gan(netDiscriminator, netGenerator):
        gen_iterations = 0
        train_best_netG_model = ''
        train_best_logprob = 0
        num_epochs_img = max(3, opt.niter / 5)
        if opt.dataset == 'lsun':
            num_epochs_img = 1
        start_time = time.time()

        for epoch in range(opt.niter):
            loss_D = 0
            loss_G = 0
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                for p in netDiscriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for p in netGenerator.parameters():  # disable grad of generator
                    p.requires_grad = False

                if opt.clamp_upper > 0:
                    # clamp parameters to a cube
                    for p in netDiscriminator.parameters():
                        p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                netDiscriminator.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                inputv.data.resize_(real_cpu.size()).copy_(real_cpu)
                label.data.resize_(batch_size).fill_(real_label)

                output = netDiscriminator(inputv)
                #errD_real = criterion(output, label)
                errD_real = torch.mean(torch.neg(torch.log(output)))
                errD_real.backward()

                # train with fake
                noisev.data.resize_(batch_size, nz, 1, 1)
                init_z(noisev.data) # totally freeze netG
                fake = netGenerator(noisev)
                label.data.fill_(fake_label)
                output = netDiscriminator(fake.detach())
                errD_fake = criterion(output, label)
                #errD_fake = torch.mean(torch.neg(torch.log(1 - torch.exp(torch.log(output)))))
                errD_fake.backward()
                errD = errD_real + errD_fake
                loss_D += errD.data.sum()
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in netDiscriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in netGenerator.parameters():  # reset requires_grad
                    p.requires_grad = True
                netGenerator.zero_grad()
                label.data.fill_(real_label)  # fake labels are real for generator cost
                init_z(noisev.data)  # totally freeze netG
                fake = netGenerator(noisev)
                fake_output = netDiscriminator(fake)
                errG = criterion(fake_output, label)
                #errG = torch.mean(torch.neg(torch.log(fake_output)))
                errG.backward()
                loss_G += errG.data.sum()
                optimizerG.step()
                gen_iterations += 1


            if epoch % num_epochs_img == 0:
                if opt.normalizeImages:
                    real_cpu = real_cpu.mul(0.5).add(0.5)

                vutils.save_image(real_cpu[0:num_images], '{0}/real_samples.png'.format(opt.experiment))
                fake = netGenerator(Variable(fixed_noise, volatile=True))
                if opt.normalizeImages:
                    fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data,
                                  '{0}/fake_samples_{1}_epoch_{2}.png'.format(opt.experiment, gen_iterations, epoch))

            # do checkpointing
            end_time = time.time()
            logger.info('[%d/%d] : Loss_D: %f Loss_G: %f, running time: %f'
                        % (epoch, opt.niter,
                           loss_D, loss_G, (end_time - start_time)))
            torch.save(netGenerator.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
            torch.save(netDiscriminator.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
            val_logprob_normalized = eval_with_KDE(netGenerator, val_set, val_noise)
            if train_best_netG_model == '' or val_logprob_normalized > train_best_logprob:
                train_best_logprob = val_logprob_normalized
                train_best_netG_model = '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch)
                logger.info(
                    'The current best model is epoch {0} with mean log probability {1}.'.format(
                        epoch, val_logprob_normalized))

        return train_best_netG_model, train_best_logprob

    def eval(netDiscriminator, netGenerator):
        gen_iterations = 0
        train_best_netG_model = ''
        train_best_logprob = 0
        num_epochs_img = max(3, opt.niter/5)
        if opt.dataset == 'lsun':
            num_epochs_img = 1
        start_time = time.time()

        for epoch in range(opt.niter):
            data_iter = iter(dataloader)
            i = 0
            loss_D = 0
            loss_G = 0
            while i < len(dataloader):
                ############################
                # (1) Update D network
                ###########################
                for p in netDiscriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for p in netGenerator.parameters():  # disable grad of generator
                    p.requires_grad = False

                Diters = 1
                if opt.D == 'wgan':
                    # train the discriminator Diters times
                    if opt.wganheuristics and (gen_iterations < 25 or gen_iterations % 500 == 0):
                        Diters = 100
                    else:
                        Diters = opt.Diters
                j = 0
                while j < Diters and i < len(dataloader):
                    j += 1

                    if opt.clamp_upper > 0:
                        # clamp parameters to a cube
                        for p in netDiscriminator.parameters():
                            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                    data = data_iter.next()
                    i += 1

                    # train with real
                    real_cpu, _ = data
                    batch_size = real_cpu.size(0)
                    netDiscriminator.zero_grad()

                    if opt.cuda:
                        real_cpu = real_cpu.cuda()
                    input.resize_as_(real_cpu).copy_(real_cpu)
                    inputv = Variable(input)

                    errD_real = netDiscriminator(inputv)
                    if opt.D == 'fgan':
                        errD_real = torch.neg(torch.log(1 + torch.exp(torch.neg(errD_real))))

                    # train with fake
                    init_z(noise.resize_(batch_size, nz, 1, 1))
                    noisev = Variable(noise)  # totally freeze netG
                    fake = Variable(netGenerator(noisev).data)
                    errD_fake = netDiscriminator(fake)
                    if opt.D == 'fgan':
                        errD_fake = torch.log(1/(1 + torch.exp(torch.neg(errD_fake))))
                        errD_fake = torch.neg(torch.log(1 - torch.exp(errD_fake)))
                    elif opt.D == 'kl':
                        errD_fake = torch.exp(errD_fake - 1)
                    loss_discriminator = torch.mean(errD_fake - errD_real)
                    loss_discriminator.backward(one)
                    optimizerD.step()
                    #loss_D += errD_fake.data + errD_real.data
                    loss_D += loss_discriminator.data


                ############################
                # (2) Update G network
                ###########################
                for p in netDiscriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in netGenerator.parameters():  # reset requires_grad
                    p.requires_grad = True
                netGenerator.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                init_z(noise.resize_(batch_size, nz, 1, 1))
                noisev = Variable(noise, volatile=False)
                fake = netGenerator(noisev)
                errG = netDiscriminator(fake)

                if opt.D == 'fgan':
                    errG = torch.neg(torch.log(1 + torch.exp(torch.neg(errG)))) # sigmoid

                errG = torch.neg(torch.mean(errG))
                errG.backward(one)
                optimizerG.step()
                loss_G += errG.data
                gen_iterations += 1

                # print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                #     % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                #     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if epoch % num_epochs_img == 0:
                if opt.normalizeImages:
                    real_cpu = real_cpu.mul(0.5).add(0.5)

                vutils.save_image(real_cpu[0:num_images], '{0}/real_samples.png'.format(opt.experiment))
                fake = netGenerator(Variable(fixed_noise, volatile=True))
                if opt.normalizeImages:
                    fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data,
                                  '{0}/fake_samples_{1}_epoch_{2}.png'.format(opt.experiment, gen_iterations, epoch))

            # do checkpointing
            end_time = time.time()
            logger.info('[%d/%d] : Loss_D: %f Loss_G: %f, running time: %f'
                        % (epoch, opt.niter,
                           loss_D[0], loss_G[0], (end_time - start_time)))
            torch.save(netGenerator.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
            torch.save(netDiscriminator.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
            val_logprob_normalized = eval_with_KDE(netGenerator, val_set, val_noise)
            if train_best_netG_model == '' or val_logprob_normalized > train_best_logprob:
                train_best_logprob = val_logprob_normalized
                train_best_netG_model = '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch)
                logger.info(
                    'The current best model is epoch {0} with mean log probability {1}.'.format(
                        epoch, val_logprob_normalized))
        return train_best_netG_model, train_best_logprob

    if opt.task == 'eval_kde':
        return sample_image_compute_density(opt.start, opt.end)
    else:
        if opt.kdeEpoch == 0:
            if opt.D == 'gan':
                best_netG_model, best_logprob = eval_gan(netD, netG)
            else:
                best_netG_model, best_logprob = eval(netD, netG)
        else:
            best_netG_model = '{0}/netG_epoch_{1}.pth'.format(opt.experiment, opt.kdeEpoch)
            print('Load model from epoch %d for KDE evaluation.' % opt.kdeEpoch)
        logger.info('Load the best model from {0} with log probability {1} .'.format(best_netG_model, best_logprob))
        netG.load_state_dict(torch.load(best_netG_model))
        if opt.cuda:
            netG.cuda()
        fake = netG(Variable(fixed_noise, volatile=True))
        if opt.normalizeImages:
            fake.data = fake.data.mul(0.5).add(0.5)
        vutils.save_image(fake.data,
                          '{0}/best_fake_samples.png'.format(opt.experiment))
        logprob_mean = eval_with_KDE(netG, test_set, test_noise)
        logger.info("On the test set, mean log probablity is %s " % logprob_mean)
        return logprob_mean



def search_hyperparams(opt):
    learn_rates = [0.0002, 0.0001, 0.00005, 0.00001]
    hidden_units = [opt.imageSize, opt.imageSize * 2, opt.imageSize * 8, opt.imageSize * 32, 1200]
    clamping_bounds = [0, 0.1, 0.01, 0.001]
    batch_size = [64, 4096]
    init_weights = ['uniform', 'xavier']
    init_noise = ['uniform_one', 'uniform_zero_one', 'gaussian']

    original_exp = opt.experiment
    error_out = open(os.path.join(original_exp, '{0}_{1}_{2}_{3}_hyperparams.error'.format(opt.dataset, opt.D, opt.A, opt.H)), 'w')
    with open(os.path.join(original_exp, '{0}_{1}_{2}_{3}_hyperparams.log'.format(opt.dataset, opt.D, opt.A, opt.H)), 'w') as out:
        max_logprob = 0
        best_config = ''
        if opt.adam:
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                opt.experiment = os.path.join(original_exp,
                                              '{0}_{1}_{2}_{3}_{4}'.format(opt.dataset, opt.D, opt.A, opt.H, lr))
                if not os.path.exists(opt.experiment):
                    os.makedirs(opt.experiment)
                try:
                    logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment,
                                                                        '{0}_{1}_{2}_{3}_evaluation.log'.format(
                                                                            opt.dataset, opt.D, opt.A, opt.H)))
                    config = '{0}\t{1}\n'.format(lr, logprob)
                    if max_logprob == 0 or logprob > max_logprob:
                        max_logprob = logprob
                        best_config = config
                    out.write(config)
                    out.flush()
                except Exception as e:
                    print(e)
                    error_out.write('learning rate {0} with error {1}'.format(lr, e))
                print(best_config)
        elif opt.A == 'dcgan' and opt.D != 'wgan':
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                for c in clamping_bounds:
                    opt.clamp_lower = -c
                    opt.clamp_upper = c
                    opt.experiment = os.path.join(original_exp, '{0}_{1}_{2}_{3}_{4}_{5}'.format(opt.dataset, opt.D, opt.A, opt.H, lr, c))
                    if not os.path.exists(opt.experiment):
                        os.makedirs(opt.experiment)
                    try:
                        logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_evaluation.log'.format(opt.dataset, opt.D, opt.A, opt.H)))
                        config = '{0}\t{1}\t{2}\n'.format(lr, c, logprob)
                        if max_logprob == 0 or logprob > max_logprob:
                            max_logprob = logprob
                            best_config = config
                        out.write(config)
                        out.flush()
                    except Exception as e:
                        print(e)
                        error_out.write('config {0} {1} with error {2}'.format(lr, c, e))
                print(best_config)
        elif opt.D == 'wgan':
            opt.clamp_lower = -0.01
            opt.clamp_upper = 0.01
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                opt.experiment = os.path.join(original_exp, '{0}_{1}_{2}_{3}_{4}'.format(opt.dataset, opt.D, opt.A, opt.H, lr))
                if not os.path.exists(opt.experiment):
                    os.makedirs(opt.experiment)
                try:
                    logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_evaluation.log'.format(opt.dataset, opt.D, opt.A, opt.H)))
                    config = '{0}\t{1}\n'.format(lr, logprob)
                    if max_logprob == 0 or logprob > max_logprob:
                        max_logprob = logprob
                        best_config = config
                    out.write(config)
                    out.flush()
                except Exception as e:
                    print(e)
                    error_out.write('learning rate {0} with error {1}'.format(lr, e))
                print(best_config)
        else:
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                for h in hidden_units:
                    opt.ngf = h
                    opt.ndf = h
                    for c in clamping_bounds:
                        opt.clamp_lower = -c
                        opt.clamp_upper = c
                        opt.experiment = os.path.join(original_exp,
                                                      '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(opt.dataset, opt.D, opt.A,
                                                                                           opt.H, lr, h, c))
                        if not os.path.exists(opt.experiment):
                            os.makedirs(opt.experiment)
                        try:

                            logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_evaluation.log'.format(opt.dataset, opt.D, opt.A, opt.H)))
                            config = '{0}\t{1}\t{2}\t{3}\n'.format(lr, h, c, logprob)
                            if max_logprob == 0 or logprob > max_logprob:
                                max_logprob = logprob
                                best_config = config
                            out.write(config)
                            out.flush()
                        except Exception as e:
                            print(e)
                            error_out.write('config {0} {1} {2} with error {3}'.format(lr, h, c, e))
                print(best_config)
        out.write('best configuration : {0}'.format(best_config))
    error_out.close()

def search_lsun_hyperparams(opt):
    learn_rates = [0.0001, 0.00005, 0.00001]
    hidden_units = [opt.imageSize, 1024]
    clamping_bounds = [0, 0.1, 0.01, 0.001]
    original_exp = opt.experiment
    error_out = open(os.path.join(original_exp, '{0}_{1}_{2}_{3}_hyperparams.error'.format(opt.dataset, opt.D, opt.A, opt.H)), 'w')
    with open(os.path.join(original_exp, '{0}_{1}_{2}_{3}_hyperparams.log'.format(opt.dataset, opt.D, opt.A, opt.H)), 'w') as out:
        max_logprob = 0
        best_config = ''
        if opt.A == 'dcgan':
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                for c in clamping_bounds:
                    opt.clamp_lower = -c
                    opt.clamp_upper = c
                    opt.experiment = os.path.join(original_exp, '{0}_{1}_{2}_{3}_{4}_{5}'.format(opt.dataset, opt.D, opt.A, opt.H, lr, c))
                    if not os.path.exists(opt.experiment):
                        os.makedirs(opt.experiment)
                    try:
                        logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_evaluation.log'.format(opt.dataset, opt.D, opt.A, opt.H)))
                        config = '{0}\t{1}\t{2}\n'.format(lr, c, logprob)
                        if max_logprob == 0 or logprob > max_logprob:
                            max_logprob = logprob
                            best_config = config
                        out.write(config)
                        out.flush()
                    except Exception as e:
                        print(e)
                        error_out.write('config {0} {1} with error {2}'.format(lr, c, e))
                print(best_config)
        else:
            for lr in learn_rates:
                opt.lrD = lr
                opt.lrG = lr
                for h in hidden_units:
                    opt.ngf = h
                    opt.ndf = h
                    for c in clamping_bounds:
                        opt.clamp_lower = -c
                        opt.clamp_upper = c
                        opt.experiment = os.path.join(original_exp,
                                                      '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(opt.dataset, opt.D, opt.A,
                                                                                           opt.H, lr, h, c))
                        if not os.path.exists(opt.experiment):
                            os.makedirs(opt.experiment)
                        try:

                            logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_evaluation.log'.format(opt.dataset, opt.D, opt.A, opt.H)))
                            config = '{0}\t{1}\t{2}\t{3}\n'.format(lr, h, c, logprob)
                            if max_logprob == 0 or logprob > max_logprob:
                                max_logprob = logprob
                                best_config = config
                            out.write(config)
                            out.flush()
                        except Exception as e:
                            print(e)
                            error_out.write('config {0} {1} {2} with error {3}'.format(lr, h, c, e))
                print(best_config)
        out.write('best configuration : {0}'.format(best_config))
    error_out.close()

def read(file_path):
    mus = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                if len(row[0]) > 0 and len(row) > 1:
                    mus.add(float(row[0]))
    return mus

def search_mu(opt, start = 0, end = 11):
    mus = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    original_exp = opt.experiment
    file_name = '{0}_{1}_{2}_{3}_{4}_{5}_search_mu.log'.format(opt.dataset, opt.D, opt.A, opt.H, opt.manualSeed, opt.critic_last_layer)
    csv_file = os.path.join(opt.experiment, file_name)
    mu_set = read(csv_file)
    with open(csv_file, 'a') as out:
        max_logprob = 0
        best_config = ''
        for i in range(start, end):
            mu = mus[i]
            if mu not in mu_set:
                opt.mu = mu
                opt.H = 'murelu'
                if mu == 1 :
                    opt.H = 'relu'
                try:
                    opt.experiment = os.path.join(original_exp, '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(opt.dataset, opt.D, opt.A, opt.H, mu, opt.manualSeed, opt.critic_last_layer))
                    if not os.path.exists(opt.experiment):
                        os.makedirs(opt.experiment)
                    logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_{4}_{5}_{6}_matsushita_mu.log'.format(opt.dataset, opt.D, opt.A, opt.H, mu, opt.manualSeed, opt.critic_last_layer)))
                    config = '{0},{1}\n'.format(mu, logprob)
                    if max_logprob == 0 or logprob > max_logprob:
                        max_logprob = logprob
                        best_config = config
                    out.write(config)
                    out.flush()
                    print('best %s ' % best_config)
                except Exception as e:
                    traceback.print_exc()
        #out.write('best configuration : {0}'.format(best_config))

def experiments_randseeds(opt, start = 0, end = 5):
    random_seeds = [1, 101, 512, 1001, 10001]
    original_exp = opt.experiment
    file_name = '{0}_{1}_{2}_{3}_{4}_experiments.csv'.format(opt.dataset, opt.D, opt.A, opt.H, opt.critic_last_layer)
    csv_file = os.path.join(opt.experiment, file_name)
    with open(csv_file, 'a') as out:
        max_logprob = 0
        best_config = ''
        for i in range(start, end):
            rand_seed = random_seeds[i]
            opt.manualSeed = rand_seed
            try:
                opt.experiment = os.path.join(original_exp, '{0}_{1}_{2}_{3}_{4}_{5}'.format(opt.dataset, opt.D, opt.A, opt.H, opt.manualSeed, opt.critic_last_layer))
                if not os.path.exists(opt.experiment):
                    os.makedirs(opt.experiment)
                logprob = train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_{4}_{5}_experiments.log'.format(opt.dataset, opt.D, opt.A, opt.H, opt.manualSeed, opt.critic_last_layer)))
                config = '{0},{1}\n'.format(rand_seed, logprob)
                if max_logprob == 0 or logprob > max_logprob:
                    max_logprob = logprob
                    best_config = config
                out.write(config)
                out.flush()
                print('best %s ' % best_config)
            except:
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | mnist')
    parser.add_argument('--subset', help='tower | bedroom')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for Critic, default=0.0001') # 0.00005
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001') # 0.00005
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--alpha', type=float, default=1, help='alpha for elu. default=1')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--D', default='kl', help='kl | gan | wgan')
    parser.add_argument('--A', default='wmlp', help='architecture : dcgan | wmlp | mlp ')
    parser.add_argument('--H', default='relu', help='activation function in hidden layers : relu | murelu | elu | ls | sp')
    parser.add_argument('--c_activation', default='none', help='activation function in the hidden layers of critic: relu | elu')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--bandwidth', type=float, default=0, help='optimal bandwidth for KDE, default=0')
    parser.add_argument('--kdeEpoch', type=int, default=0, help='The epoch of the model that is loaded for KDE evaluation.')
    parser.add_argument('--weightDecay', type=float, default=0)
    parser.add_argument('--task', default='train', help='rand')
    parser.add_argument('--init_z', default='gaussian', help='uniform_one | uniform_zero_one | gaussian')
    parser.add_argument('--init_w', default='xavier', help='xavier | uniform')
    parser.add_argument('--normalizeImages', type=bool, default=False)
    parser.add_argument('--last_layer', default='sigmoid', help='none | sigmoid | tanh')
    parser.add_argument('--critic_last_layer', default='none', help='none | sigmoid | tanh | matsu')
    parser.add_argument('--mu', type=float, default=0, help='mu for matsushita')
    parser.add_argument('--manualSeed', type=int, default=512, help='random seed')
    parser.add_argument('--wganheuristics', type=bool, default=False)
    parser.add_argument('--kde_result_dir', default='', help='folder which stores the images and KDE results created by the generator')
    parser.add_argument('--start', type=int, default=0,
                        help='The starting epoch of the model that is loaded for KDE evaluation.')
    parser.add_argument('--end', type=int, default=99,
                        help='The last epoch of the model that is loaded for KDE evaluation.')
    opt = parser.parse_args()


    # mnist

    opt.cuda = True
    if opt.dataset == 'mnist':
        opt.imageSize = 32
        opt.last_layer = 'sigmoid'
        opt.niter = 100
    elif opt.dataset == 'lsun':
        opt.imageSize = 64
        opt.last_layer = 'tanh'

    if opt.A == 'mlp':
        if opt.D == 'gan':
            opt.lrD = 0.0002
            opt.lrG = 0.0002
            opt.c_activation = 'elu'
            opt.init_w = 'xavier'
            opt.init_z = 'uniform_zero_one'
            opt.batchSize = 64
            opt.clamp_lower = 0
            opt.clamp_upper = 0
            opt.adam = True
            opt.ndf = 1024
            opt.ngf = 1024
            if opt.critic_last_layer == 'none':
                opt.critic_last_layer = 'sigmoid'

        elif opt.D == 'wgan':
            opt.lrD = 0.0002
            opt.lrG = 0.0002
            opt.c_activation = 'elu'
            opt.init_w = 'xavier'
            opt.init_z = 'uniform_zero_one'
            opt.batchSize = 64
            opt.clamp_lower = -0.01
            opt.clamp_upper = 0.01
            opt.adam = False
            opt.ndf = 1024
            opt.ngf = 1024
    elif opt.A == 'dcgan':
        if opt.D == 'gan':
            opt.lrD = 0.0002
            opt.lrG = 0.0002
            opt.init_z = 'gaussian'
            opt.batchSize = 64
            opt.clamp_lower = 0
            opt.clamp_upper = 0
            opt.adam = True
            if opt.critic_last_layer == 'none':
                opt.critic_last_layer = 'sigmoid'
        elif opt.D == 'wgan':
            opt.lrD = 0.0002
            opt.lrG = 0.0002
            opt.init_z = 'gaussian'
            opt.batchSize = 64
            opt.clamp_lower = -0.01
            opt.clamp_upper = 0.01
            opt.adam = False

    if opt.cuda:
        if opt.gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    if opt.task == 'train':
        train(opt=opt, log_file_path=os.path.join(opt.experiment, '{0}_{1}_{2}_{3}_{4}_{5}.log'.format(opt.D, opt.A, opt.H, opt.c_activation, opt.manualSeed, opt.critic_last_layer)))
    elif opt.task == 'eval_kde':
        print('max log prob is %s ' % train(opt=opt, log_file_path=os.path.join(opt.kde_result_dir, '{0}_{1}_{2}.log'.format(opt.D, opt.A, opt.H))))
    elif opt.task == 'mu':
        search_mu(opt=opt)
    elif opt.task == 'murange':
        search_mu(opt=opt, start = opt.start, end = opt.end)
    elif opt.task == 'rand':
        experiments_randseeds(opt=opt, start=opt.start, end=opt.end)
    else:
        assert opt.task == 'hyper'
        if opt.dataset == 'lsun':
            search_lsun_hyperparams(opt=opt)
        else:
            search_hyperparams(opt=opt)


if __name__ == '__main__':
    main()