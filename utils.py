import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt


class cifar(nn.Module):
    '''
    Model used to generate adversarial CIFAR10 images.
    This model achieves 73.9% test set accuracy.

    NOTE: This model was trained with dropout_prob=0.1
    '''
    def __init__(self, dropout_prob):
        super(cifar, self).__init__()
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(40, 40, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(40, 20, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv2(self.conv1(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob, training=self.training), 2)

        x = F.relu(self.conv4(self.conv3(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob, training=self.training), 2)

        x = F.relu(self.conv6(self.conv5(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob, training=self.training), 2)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



class mnist(nn.Module):
    '''
    Model used to generate adversarial MNIST images.
    This model achieves 99.1% test set accuracy.

    NOTE: This model was trained with dropout_prob=0.1
    '''
    def __init__(self, dropout_prob):
        super(mnist, self).__init__()
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv2(self.conv1(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob, training=self.training), 2)

        x = F.relu(self.conv4(self.conv3(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob, training=self.training), 2)

        x = x.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#================================================================


def trained_model(dataset, dropout_prob=0.1, path_to_pretrained=None):
    '''
    Return a pre-trained model corresponding to either CIFAR10 or MNIST
    Arguments:
    - dataset: string equal to 'cifar' or 'mnist' specifying the type of model
    - path_to_pretrained: a string indicating path to file where pre-trained 
                          parameter weights are stored
    '''
    assert (dataset == 'cifar' or dataset == 'mnist'), \
                'Model must either be \'cifar\' or \'mnist\''

    if path_to_pretrained is None:
        if dataset == 'cifar':
            path_to_pretrained = 'models/original/saved_cifar_model'
        if dataset == 'mnist':
            path_to_pretrained = 'models/original/saved_mnist_model'

    if dataset == 'cifar':
        # Initialize model
        model = cifar(dropout_prob=dropout_prob)
        # Load pre-trained parameters
        model.load_state_dict(torch.load(path_to_pretrained)['state_dict'])
        return model

    if dataset == 'mnist':
        # Initialize model
        model = mnist(dropout_prob=dropout_prob)
        # Load pre-trained parameters
        model.load_state_dict(torch.load(path_to_pretrained)['state_dict'])
        return model



def get_data(dataset, _train=True, _transforms=transforms.ToTensor(), _batch_size=50):
    '''
    Returns a Pytorch dataloader for the dataset specified by argument dataset
    Arguments:
    - dataset: string equal to 'cifar' or 'mnist' specifying the type of data to load 
    - _train: Boolean indicating whether function should return train or test data
    - _trainsforms: torchvision transformations to apply to dataset
    - _batch_size: integer specifying size of batches in dataset

    NOTE: this function assumes that the CIFAR and MNIST dataset directories
          are in the same directory as this script
    '''

    assert (dataset == 'cifar' or dataset == 'mnist'), \
                'Dataset must either be \'cifar\' or \'mnist\''

    if dataset == 'cifar':
        data = torchvision.datasets.CIFAR10('data/', train=_train, transform=_transforms, download=True)

    if dataset == 'mnist':
        data = torchvision.datasets.MNIST('data/', train=_train, transform=_transforms, download=True)

    dataloader = DataLoader(data, batch_size=_batch_size, shuffle=True, num_workers=1)
    return dataloader



class Adversary_Data(Dataset):
    '''
    A customized data set for adversarial images
    '''
    def __init__(self, file_name, original_label=None, target_label=None, num_examples=None):
        '''
        Intialize the adversarial image dataset
        Args:
        - file_name: name of file where adversarial data is stored
        - original_label: the original label of images
        - target_label: the label that adversaries trick model into predicting
        '''
        if '.npz' not in file_name:
            file_name += '.npz'
        data = np.load(file_name)
        
        original_labels = data['original_labels']
        target_labels = data['target_labels']

        # Subset the data
        if original_label is not None and target_label is not None:
            assert (original_label != target_label), \
                'Target label must be different from original label!'
            mask = (original_labels == original_label) * (target_labels == target_label)
        elif original_label is not None:
            mask = (original_labels == original_label)
        elif target_label is not None:
            mask = (target_labels == target_label)
        else:
            mask = np.ones_like(original_labels, dtype=bool)

        self.original_labels = torch.from_numpy(original_labels[mask])
        self.target_labels = torch.from_numpy(target_labels[mask])
        self.originals = torch.from_numpy(data['original_images'][mask])
        self.perturbations = torch.from_numpy(data['perturbations'][mask])
        self.adversaries = torch.from_numpy(data['adversarial_images'][mask])

        if num_examples is not None:
            self.original_labels = self.original_labels[:num_examples]
            self.target_labels = self.target_labels[:num_examples]
            self.originals = self.originals[:num_examples]
            self.perturbations = self.perturbations[:num_examples]
            self.adversaries = self.adversaries[:num_examples]
        
        self.len = self.originals.size(0)
        
    def __getitem__(self, index):
        '''
        Get a sample from the dataset
        '''
        original = self.originals[index]
        pert = self.perturbations[index]
        adv = self.adversaries[index]
        orig_label = self.original_labels[index].item()
        target_label = self.target_labels[index].item()
        return original, pert, adv, orig_label, target_label

    def __len__(self):
        '''
        Total number of samples in the dataset
        '''
        return self.len


def get_adv_data(file_name, original_label=None, target_label=None, num_examples=None, batch_size=50):
    '''
    Returns a Pytorch dataset object containing data from an adversarial 
    dataset stored at file specified by file_name
    Args:
    - file_name: name of file where adversarial data is stored
    - original_label: the original label of images
    - target_label: the label that adversaries trick model into predicting
    - batch_size: size of batch
    '''

    data = Adversary_Data("data/" + file_name, original_label, target_label, num_examples)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    return dataloader



