""" helper function

author baiyu
"""
# hacky override for pickling lambda functions
import dill as pickle

import os
import sys
import re
import datetime

import numpy

from conf import settings
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cub_data_loader import *



def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16' and (args.dataset == 'MNIST' or args.dataset == 'FashionMNIST'):
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=10)
    elif args.net == 'vgg16' and args.dataset == 'CIFAR10':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=10, linear_size=32768)    
    elif args.net == 'vgg16' and args.dataset == 'CIFAR100':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=100,linear_size=32768)
    elif args.net == 'vgg16' and args.dataset == 'CUB':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=200,linear_size=32768)
    elif args.net == 'vgg16' and args.dataset == 'Food101':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=101,linear_size=32768)
    elif args.net == 'vgg11' and (args.dataset == 'MNIST' or args.dataset == 'FashionMNIST'):
        from models.vgg import vgg11_bn
        net = vgg11_bn(10)
    elif args.net == 'vgg11' and args.dataset == 'CIFAR10':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=10, linear_size=32768)  
    elif args.net == 'vgg11' and args.dataset == 'CIFAR100':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=100,linear_size=32768)
    elif args.net == 'vgg11' and args.dataset == 'CUB':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=200,linear_size=32768)
    elif args.net == 'vgg11' and args.dataset == 'Food101':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=101,linear_size=32768)
    elif args.net == 'resnet18' and (args.dataset == 'MNIST' or args.dataset == 'FashionMNIST'):
        from models.resnet import resnet18
        net = resnet18(num_classes=10)
    elif args.net == 'resnet18' and args.dataset == 'CIFAR10':
        from models.vgg import resnet18
        net = resnet18(num_classes=10, linear_size=32768)  
    elif args.net == 'resnet18' and args.dataset == 'CIFAR100':
        from models.resnet import resnet18
        net = resnet18(num_classes=100,linear_size=32768)
    elif args.net == 'resnet18' and args.dataset == 'CUB':
        from models.resnet import resnet18
        net = resnet18(num_classes=200,linear_size=32768)
    elif args.net == 'resnet18' and args.dataset == 'Food101':
        from models.resnet import resnet18
        net = resnet18(num_classes=101,linear_size=32768)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(args, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        transform_train = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    elif "Caltech" in args.dataset:
        transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ]) 
    else: 
        transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    if args.dataset == "MNIST":
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
        mnist_training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        training_loader = DataLoader(
        mnist_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "FashionMNIST":
        fashionmnist_training = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        training_loader = DataLoader(
        fashionmnist_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CIFAR10":
        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CIFAR100":
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "Food101":
        caltech256_training = torchvision.datasets.Food101(root='./data', download=True, transform=transform_train)
        training_loader = DataLoader(
        caltech256_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CUB":
        cub_training = Cub2011(root='./data', train=True, transform=transform_train, download=True)
        training_loader = DataLoader(
        cub_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return training_loader


def get_test_dataloader(args, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        transform_test = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    elif "Caltech" in args.dataset:
        transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ]) 
    else: 
        transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    
    if args.dataset == "MNIST":
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(
        mnist_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "FashionMNIST":
        fashionmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(
        fashionmnist_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CIFAR10":
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CIFAR100":
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "Food101":
        caltech256_test = torchvision.datasets.Food101(root='./data', split="test", download=True, transform=transform_test)
        test_loader = DataLoader(
        caltech256_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif args.dataset == "CUB":
        cub_test = Cub2011(root='./data', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(
        cub_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]