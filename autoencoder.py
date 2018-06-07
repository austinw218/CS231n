import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    
    def __init__(self, dataset_name,dropout_prob):
        super().__init__()
        self.dataset = dataset_name
        self.dropout = dropout_prob
        if dataset_name =="mnist":
            self.num_channels = 1
            self.size = 28

        if dataset_name == "cifar":
            self.num_channels = 3
            self.size = 32
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(self.num_channels, 16, kernel_size=3)
        self.enc_cnn_2 = nn.Conv2d(16, 10, kernel_size=3)


        # Activation/max pooling
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.unpool = nn.MaxUnpool2d(2)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.batch_norm_2 = nn.BatchNorm2d(10)
        self.batch_norm_3 = nn.BatchNorm2d(8)
        self.batch_norm_4 = nn.BatchNorm2d(self.num_channels)

        # Decoder specification
        self.dec_cnn_1 = nn.ConvTranspose2d(10, 8, kernel_size=3)
        self.dec_cnn_2 = nn.ConvTranspose2d(8, self.num_channels, kernel_size = 3)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        out = F.sigmoid(out) # Make output between 0 and 1

        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = self.relu(code)
        code = F.dropout(code, self.dropout, training=self.training)
        code = self.batch_norm_1(code)

        code = self.enc_cnn_2(code)
        code = self.relu(code)
        code = F.dropout(code, self.dropout, training=self.training)
        code = self.batch_norm_2(code)

        return code
    
    def decode(self, code):
        out = self.dec_cnn_1(code)
        out = self.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.batch_norm_3(out)

        out = self.dec_cnn_2(out)
        out = self.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.batch_norm_4(out)

        return out
