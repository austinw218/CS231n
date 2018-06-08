import torch
import utils
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
import stats
import autoencoder
import time

from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms


# Functionality to save model parameters
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

# Autoencoder test

if __name__ == "__main__":
    start = time.time()

    # Hyperparameters
    num_epochs = 15
    batch_size = 128
    lr = 0.002
    optimizer_cls = optim.Adam

    # Data loader
    loader = utils.get_data('mnist', _train=True, _transforms=transforms.ToTensor(), 
                            _batch_size=batch_size)
    test_loader = utils.get_data('mnist', _train=False, _transforms=transforms.ToTensor(), 
                                 _batch_size=batch_size)



    # Instantiate model
    auto = autoencoder.AutoEncoder('mnist',0.1)
    loss_fn = nn.BCELoss()
    optimizer = optimizer_cls(auto.parameters(), lr=lr)

    num_batches = 150

    # Training loop
    for epoch in range(num_epochs):
        print('Epoch: {}. Time elapsed: {:.1f} minutes'.format(epoch, (time.time() - start)/60.0) )
        
        batch_num = 0
        auto.train()
        
        for i, (orig, _) in enumerate(loader):
            
            if batch_num > num_batches:
                break
            
            batch_num += 1

            out, code = auto(orig)
            
            optimizer.zero_grad()
            loss = loss_fn(out, orig)
            loss.backward()
            optimizer.step()

    save_checkpoint('mnist_autoencoder_model',auto,optimizer)

