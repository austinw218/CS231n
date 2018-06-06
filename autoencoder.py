import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset = dataset_name

        if dataset_name =="mnist":
            self.num_channels = 1
            self.size = 28

        if dataset_name == "cifar":
            self.num_channels = 3
            self.size = 32
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(self.num_channels, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 8, kernel_size=5)

        # Decoder specification
        self.dec_cnn_1 = nn.ConvTranspose2d(8, 6, kernel_size = 5)
        self.dec_cnn_2 = nn.ConvTranspose2d(6, self.num_channels, kernel_size = 5)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        out = F.sigmoid(out) # Make output between 0 and 1

        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = self.enc_cnn_2(code)
        return code
    
    def decode(self, code):
        out = self.dec_cnn_1(code)
        out = self.dec_cnn_2(out)
        return out
    

# Load data
#train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor())
#test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
#train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)


# Instantiate model
'''
autoencoder = AutoEncoder(code_size)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
    
    for i, (images, _) in enumerate(train_loader):    # Ignore image labels
        out, code = autoencoder(Variable(images))
        
        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])
'''


