# from glob import glob
from PIL import Image
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import random



# Class for calculating various statistics for the perturbation arrays.
class Stats:

    def __init__(self, file_name):

        if '.npz' not in file_name:
            file_name += '.npz'
        data = np.load('data/' + file_name)

        self.true_labels = data['original_labels']
        self.adv_labels = data['target_labels']
        self.originals = data['original_images'].transpose((0,2,3,1))
        self.perturbations = data['perturbations'].transpose((0,2,3,1))
        self.images = data['adversarial_images'].transpose((0,2,3,1))
    
    # Displays a specific adversarial histogram 
    def histogram(self, orig_label=None, adv_label=None, content="pert", style="pixel", show=True, ax=None):
        perturbations = self.subset(content,orig_label,adv_label)
        if style == "pixel":
            histogram = perturbations.flatten()
            bins = np.arange(-255,256)
        elif style == "l1":
            n = perturbations.shape[0]
            histogram = np.linalg.norm(perturbations.reshape((n,-1)), ord=1, axis=1)
            bins = None
        elif style == "l2":
            n = perturbations.shape[0]
            histogram = np.linalg.norm(perturbations.reshape((n,-1)), ord=2, axis=1)
            bins = None
        elif style == "max":
            n = perturbations.shape[0]
            histogram = np.max(perturbations.reshape((n,-1)), axis=1)
            bins = None
        if show:
            plt.hist(histogram, bins, density=True)
            title = {"pert": "Perturbations",
                     "img": "Images",
                     "pixel": "Pixel",
                     "l2": "L2 Norm",
                     "l1": "L1 Norm",
                     "max": "Max"}
            plt.suptitle("Adversarial {} ({}): {} to {}".format(title[content],title[style],
                                                                orig_label,adv_label))
            plt.show()
        else:
            ax.hist(histogram, bins, density=True)
    
    # Display grid of all adversarial histograms
    def histogram_grid(self, content="pert", style="pixel"):
        fig, ax = plt.subplots(10, 10, sharex='col', sharey='row', figsize=(20, 20))
        for i in range(10):
            for j in range(10):
                if i != j:
                    self.histogram(i, j, content, style, False, ax[i, j])
                if i == 9:
                    ax[i, j].set_xlabel("{0:d}".format(j + 1))
                if j == 0:
                    ax[i, j].set_ylabel("{0:d}".format(i + 1))
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.text(0.5, 0.04, 'Adversarial', ha='center')
        fig.text(0.04, 0.5, 'Original', va='center', rotation='vertical')
        title = {"pert": "Perturbations",
                 "img": "Images",
                 "pixel": "Pixel",
                 "l2": "L2 Norm",
                 "l1": "L1 Norm"}
        fig.suptitle("Adversarial {} ({})".format(title[content],title[style]))
        plt.show()
    
    # Displays a specific adversarial image     
    def image(self, orig_label, adv_label, content="pert", style="average", show=True, ax=None):
        data = self.subset(content,orig_label,adv_label)
        if style == "random":
            n = data.shape[0]
            image = data[random.randint(0,n),:,:,:].squeeze()
        elif style == "average":
            image = np.mean(data, axis=0).squeeze()
        if show:
            title = {"pert": "Perturbation",
                     "img": "Image",
                     "orig": "Original",
                     "random": "Random",
                     "average": "Average"}
            plt.imshow(image)#, cmap="Greys")
            plt.colorbar()
            plt.suptitle("Adversarial {} ({}): {} to {}".format(title[content],title[style],
                                                                orig_label,adv_label))
            plt.show()
        else:
            ax.imshow(image)#, cmap="Greys")
    
    # Display grid of images
    def image_grid(self, content="pert", style="average"):
        fig, ax = plt.subplots(10, 10, sharex='col', sharey='row', figsize=(20, 20))
        for i in range(10):
            for j in range(10):
                self.image(i, j, content, style, False, ax[i, j])
                if i == 9:
                    ax[i, j].set_xlabel("{0:d}".format(j + 1))
                if j == 0:
                    ax[i, j].set_ylabel("{0:d}".format(i + 1))
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.text(0.5, 0.04, 'Adversarial', ha='center')
        fig.text(0.04, 0.5, 'Original', va='center', rotation='vertical')
        title = {"pert": "Perturbations",
                 "img": "Images",
                 "orig": "Original",
                 "random": "Random",
                 "average": "Average"}
        fig.suptitle("Adversarial {} ({})".format(title[content],title[style]))
        plt.show()
    
    
    # Print summary statistics for a specific adversarial attack      
    def summary(self, orig_label, adv_label, content="pert", stat="all"):
        data = self.subset(content,orig_label,adv_label)
        mean = np.mean(data)
        std = np.std(data)
        mn = np.min(data)
        mx = np.max(data)
        if stat == "mean" or stat == "all":   
            print('mean: {0}'.format(mean))
        if stat == "std" or stat == "all":
            print('std: {0}'.format(std))
        if stat == "min" or stat == "all":
            print('min: {0}'.format(mn))
        if stat == "max" or stat == "all":
            print('max: {0}'.format(mx))
            
    
    def subset(self, content, true_label, adv_label):
        """

        Retrieves the perturbations with the given true label and given adversarial label.

        :param true_label: The label of the image used to generate the perturbation. Default
                           is None; if no argument is supplied, no filter on the true labels
                           will be applied

        :param adv_label: The label of the adversarial image caused by the perturbation. 
                          Default is None; if no argument is supplied, no filter on the 
                          adversarial labels will be applied

        Returns a filtered numpy array of dimension (N', C, H, W), consisting of the
        perturbations with the specified true_label and adv_label
        """
        # Declare Mask     
        if true_label == None and adv_label == None:
            mask = np.ones(self.adv_labels.shape, dtype=bool)
        elif true_label == None:
            mask = self.adv_labels == adv_label 
        elif adv_label == None:
            mask = self.true_labels == true_label
        else:
            mask = np.logical_and(self.adv_labels == adv_label, self.true_labels == true_label)
        # Apply Mask
        if content == "pert":
            return self.perturbations[mask,:,:,:]
        if content == "img":
            return self.images[mask,:,:,:]
        if content == "org":
            return self.originals[mask,:,:,:]
