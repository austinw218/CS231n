
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


class cifar(nn.Module):
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
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob), 2)

        x = F.relu(self.conv4(self.conv3(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob), 2)

        x = F.relu(self.conv6(self.conv5(x)))
        x = F.max_pool2d( F.dropout(x, p=self.dropout_prob), 2)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_prob)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



# Functionality to save model parameters
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)



def train(num_epoch, 
          model, 
          optimizer, 
          trainset_loader, 
          testset_loader, 
          saved_model_file_name, 
          log_interval=100):
    
    start = time.time()
    best_accuracy = 0.0
    iteration = 0
    
    for ep in range(num_epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            model.train()  # set training mode
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                print('')
                print('Total time elapsed (in minutes): {:.2f}'.format( (time.time() - start)/60. ))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
                
        # Evaluate train/test accuracy
        #----------------------------------------------------------------------------
        model.eval()
        test_correct = 0
        train_correct = 0
        with torch.no_grad():
            # Test
            for data, target in testset_loader:
                output = model(data)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_accuracy = test_correct/len(testset_loader.dataset)

            # Train
            for data, target in trainset_loader:
                output = model(data)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_accuracy = train_correct/len(trainset_loader.dataset)

        print('Test set: Accuracy: {}/{} ({:.1f}%)'.format(
            test_correct, len(testset_loader.dataset), 100. * test_accuracy ))
        print('Train set: Accuracy: {}/{} ({:.1f}%)'.format(
            train_correct, len(trainset_loader.dataset), 100. * train_accuracy ))
        
        # Save best model
        if test_accuracy > best_accuracy:
            save_checkpoint('{}{}_best_model'.format('cifar_saved_models/',saved_model_file_name), model, optimizer)
            best_accuracy = test_accuracy
        #----------------------------------------------------------------------------


    # Save model after final epoch
    save_checkpoint('{}{}_final_model'.format('cifar_saved_models/', saved_model_file_name), model, optimizer)
    



if __name__ == '__main__':
    cifar_train = torchvision.datasets.CIFAR10('data/',
                                               train=True,
                                               transform=transforms.ToTensor())

    cifar_test = torchvision.datasets.CIFAR10('data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    trainset_loader = DataLoader(cifar_train, batch_size=50, shuffle=True, num_workers=1)
    testset_loader = DataLoader(cifar_test, batch_size=50, shuffle=True, num_workers=1)

    model = cifar(dropout_prob=0.1)
    optimizer = optim.Adam(model.parameters())

    num_epoch = 10
    # Train model and save trained parameters

    train(num_epoch, 
          model, 
          optimizer, 
          trainset_loader, 
          testset_loader, 
          'cifar_model',
          log_interval=100)





