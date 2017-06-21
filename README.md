# f-GANs in an Information Geometric Nutshell

Pytorch implementation of [f-GANs in an Information Geometric Nutshell](http://arxiv.org/abs/)

## Prerequisites

- Python 2.7
- [Pytorch 0.1.12](http://pytorch.org)
- [numpy 1.12.1](http://www.numpy.org/)


## Usage

Put both mnist and lsun in a folder called DATA_ROOT. Download lsun with https://github.com/fyu/lsun. MNIST will be downloaded automatically in the first run.

    $ python download.py -o <DATA_ROOT> -c tower


Assume all experimental results are put in EXPERIMENTAL_RESULTS.

Evaluate a feedforward network with wasserstein GAN loss and mu-ReLU as the activation of hidden layers of the generator:

    $ python main.py --dataset mnist --dataroot <DATA_ROOT> --cuda -D wgan -A mlp -H murelu --experiment <EXPERIMENTAL_RESULTS> --task mu

Evaluate DCGAN with GAN as the loss, and mu-ReLU as the activation of hidden layers of the generator:

    $ python main.py --dataset lsun --subset tower --dataroot <DATA_ROOT> --cuda -D gan -A dcgan -H murelu --experiment <EXPERIMENTAL_RESULTS> --task mu


## Author

Lizhen Qu / [@qulizhen](https://cecs.anu.edu.au/people/lizhen-qu)