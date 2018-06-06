import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out
    

IMAGE_SIZE = 784
IMAGE_WIDTH = IMAGE_HEIGHT = 28

# Hyperparameters
code_size = 20
num_epochs = 5
batch_size = 128
lr = 0.002
optimizer_cls = optim.Adam

# Load data
train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor())
test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# Instantiate model
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