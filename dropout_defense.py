import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import utils
from matplotlib import gridspec
import seaborn


class Dropout_defense:
    '''
    Class that helps experiment with dropout defense strategy
    '''
    def __init__(self, 
                 dataset, 
                 file_name=None,
                 num_examples=None):
        '''
        Arguments:
        - dataset: 'cifar' or 'mnist'
        - dropout_prob: probability of dropping nodes in model forward pass
        - file_name: the name of the file where the adversarial data is
                     stored (this is a .npz file)
        - original_label: original label of images
        - target_label: label of class the model was tricked into predicting
        '''

        if file_name is None:
            if dataset == 'cifar':
                file_name = 'data/cifar_eps_5e-3'
            if dataset == 'mnist':
                file_name = 'data/mnist_eps_5e-3_norm_2_num_iters_50.npz'
        self.dataset = dataset
        self.file_name = file_name

        # Load model
        self.model = utils.trained_model(self.dataset)
        self.model.train() # put model in train mode to enable dropout

        # Load data
        start = time.time()
        print('Loading data...')
        print('  | 0 1 2 3 4 5 6 7 8 9')
        print('-----------------------')
        self.data = dict()
        for orig in range(10):
            print('{} | '.format(orig), end='')
            for adv in range(10):
                if adv == orig:
                    print('  '.format(), end='')
                    continue
                print('1 '.format(), end='')
                # print('Loading pair ({},{})'.format(orig,adv))
                self.data[(orig,adv)] = utils.get_adv_data(self.file_name, 
                                                           original_label=orig, 
                                                           target_label=adv,
                                                           num_examples=num_examples,
                                                           batch_size=1)
            print('')
        print('Data loaded. Took {:.1f} seconds.'.format(time.time() - start))


    def reset_model(self, dropout_prob):
        '''
        Initialize a new model to use with object's data. This is used
        to reset the dropout probability parameter.
        '''
        self.model = utils.trained_model(self.dataset, dropout_prob)
        self.model.train()


    def ensemble_forward_pass(self, image, ensemble_size):
        '''
        Perform forward pass multiple times, with dropout enabled, for 
        one single image
        Arguments:
        - image: a pytorch tensor containing data for one single example. 
                 Should be of shape (1,C,H,W), where C = channels, 
                 H = height, W = width
        - ensemble_size: the number of forward passes to perform
        '''
        image_set = image.clone().repeat(ensemble_size,1,1,1)
        output = self.model(image_set)
        return output.detach()

    def filter_accuracy(self, 
                        dropout_prob, 
                        ensemble_size,
                        original_label, 
                        target_label):
        '''
        Computes the percentage of adversarial images that were successfully
        thwarted
        Arguments:
        -
        '''
        # Reset model to have proper dropout probability
        self.reset_model(dropout_prob=dropout_prob)

        num_corrected = 0
        num_fooled = 0

        data = self.data[(original_label, target_label)]
        for (i, data_tuple) in enumerate(data):
            original, pert, adv, orig_label, target_label = data_tuple # unpack
            ensemble = torch.argmax(self.ensemble_forward_pass(
                                        adv, ensemble_size), dim=1)
            counts = np.eye(10)[ensemble.numpy()].sum(axis=0)
            pred = np.argmax(counts)
            if pred == original_label:
                num_corrected += 1
            if pred == target_label:
                num_fooled += 1

        return (num_corrected/len(data), num_fooled/len(data))


    def filter_heatmap(self, dropout_prob, ensemble_size):
        '''
        Visualize how well dropout thwarts adversarial images for each
        (original, target) pair
        '''
        corrected_array = np.zeros((10,10))
        fooled_array = np.zeros((10,10))

        start = time.time()
        print('  | 0 1 2 3 4 5 6 7 8 9')
        print('-----------------------')
        for orig in range(10):
            print('{} | '.format(orig), end='')
            for target in range(10):
                if target == orig:
                    print('  '.format(), end='')
                    continue
                print('1 '.format(), end='')
                corrected, fooled = self.filter_accuracy(dropout_prob=dropout_prob,
                                                         ensemble_size=ensemble_size,
                                                         original_label=orig,
                                                         target_label=target)
                corrected_array[orig,target] = corrected
                fooled_array[orig,target] = fooled
            print('')
        print('Took {:.2f} minutes'.format( (time.time() - start)/60. ))

        self.corrected_array = corrected_array
        self.fooled_array = fooled_array

        _min = min(corrected_array.min(), fooled_array.min())
        _max = max(corrected_array.max(), fooled_array.max())

        seaborn.heatmap(self.corrected_array, linewidth=0.5, cmap='hot', vmin=_min, vmax=_max, mask=np.eye(10,10))
        plt.title('Percentage Corrected')
        plt.xlabel('Target Label')
        plt.ylabel('Original Label')
        plt.show()

        seaborn.heatmap(self.fooled_array, linewidth=0.5, cmap='hot', vmin=_min, vmax=_max, mask=np.eye(10,10))
        plt.title('Percentage Fooled')
        plt.xlabel('Target Label')
        plt.ylabel('Original Label')
        plt.show()


    def get_uncertainty_score(self, ensemble_output, method):
        '''
        Computes the uncertainty score for a single image
        Arguments:
        - ensemble_output: the output from the set of forward passes for a
                           single example. This will be a set of n softmax 
                           probabilities, where n is the ensemble size.
        '''
        preds = torch.argmax(ensemble_output, dim=1).numpy()

        if method == 'variance':
            return preds.var()

        if method == 'entropy':
            counts = np.eye(10)[preds].sum(axis=0)
            probs = counts/counts.sum()
            return (probs * np.log(probs)).sum()


    def uncertainty(self, ensemble_size, method, original_label, target_label):
        '''
        Returns the average uncertainty score for the input (original_label, target_label) pair
        '''
        orig_uncertainty = list()
        adv_uncertainty = list()

        data = self.data[(original_label, target_label)]
        for (i, data_tuple) in enumerate(data):
            original, pert, adv, orig_label, target_label = data_tuple # unpack

            orig_output = self.ensemble_forward_pass(original, ensemble_size)
            adv_output = self.ensemble_forward_pass(adv, ensemble_size)

            orig_uncertainty.append(self.get_uncertainty_score(orig_output,method=method))
            adv_uncertainty.append(self.get_uncertainty_score(adv_output,method=method))

        return (np.array(orig_uncertainty).mean(), np.array(adv_uncertainty).mean())


    def uncertainty_heatmap(self, dropout_prob, ensemble_size, method):
        '''
        Visualize how well dropout thwarts adversarial images for each
        (original, target) pair
        '''
        # Reset model to have proper dropout probability
        self.reset_model(dropout_prob=dropout_prob)

        orig_array = np.zeros((10,10))
        adv_array = np.zeros((10,10))

        start = time.time()
        print('  | 0 1 2 3 4 5 6 7 8 9')
        print('-----------------------')
        for orig in range(10):
            print('{} | '.format(orig), end='')
            for target in range(10):
                if target == orig:
                    print('  '.format(), end='')
                    continue
                print('1 '.format(), end='')
                orig_score, adv_score = self.uncertainty(ensemble_size=ensemble_size,
                                                         method=method,
                                                         original_label=orig,
                                                         target_label=target)
                orig_array[orig,target] = orig_score
                adv_array[orig,target] = adv_score
            print('')
        print('Took {:.2f} minutes'.format( (time.time() - start)/60. ))

        self.orig_score_array = orig_array
        self.adv_score_array = adv_array

        _min = min(orig_array.min(), adv_array.min())
        _max = max(orig_array.max(), adv_array.max())

        seaborn.heatmap(self.orig_score_array, linewidth=0.5, cmap='hot', vmin=_min, vmax=_max, mask=np.eye(10,10))
        plt.title('Average Uncertainty Score of Originals')
        plt.xlabel('Target Label')
        plt.ylabel('Original Label')
        plt.show()

        seaborn.heatmap(self.adv_score_array, linewidth=0.5, cmap='hot', vmin=_min, vmax=_max, mask=np.eye(10,10))
        plt.title('Average Uncertainty Score of Adversaries')
        plt.xlabel('Target Label')
        plt.ylabel('Original Label')
        plt.show()



    def visualize(self, 
                  dropout_prob, 
                  original_label, 
                  target_label,
                  ensemble_size,
                  type='prediction', 
                  num_to_plot=25):
        '''
        Plots ensemble outputs for a random batch of data.
        Arguments:
        - type: whether to plot a histogram of predictions or of the
                cumulative probability mass for each class across the ensemble

        '''

        # Reset model to have proper dropout probability
        self.reset_model(dropout_prob=dropout_prob)

        for (i, data_tuple) in enumerate(self.data[(original_label,target_label)]):
            if i >= num_to_plot:
                break
            original, pert, adv, orig_label, target_label = data_tuple # unpack

            if type == 'prediction':
                original_ensemble = torch.argmax(self.ensemble_forward_pass(
                                                 original, ensemble_size), dim=1)
                adv_ensemble = torch.argmax(self.ensemble_forward_pass(
                                            adv, ensemble_size), dim=1)
                original_counts = np.eye(10)[original_ensemble.numpy()].sum(axis=0)
                adv_counts = np.eye(10)[adv_ensemble.numpy()].sum(axis=0)
                self.plot_image(original, adv, 
                                original_counts/original_counts.sum(), adv_counts/adv_counts.sum(), 
                                original_label, target_label)

            if type == 'probability':
                original_ensemble = self.ensemble_forward_pass(original, ensemble_size).numpy()
                adv_ensemble = self.ensemble_forward_pass(adv, ensemble_size).numpy()
                self.plot_image(original, adv, 
                                np.exp(original_ensemble).sum(axis=0), np.exp(adv_ensemble).sum(axis=0), 
                                original_label, target_label)


    def plot_image(self, 
                   original, adversary, 
                   original_output, adv_output, 
                   original_label, target_label):
        '''
        Plots ensemble outputs for a single image juxtaposed with the original
        and adversarial image.
        Arguments:
        - original: torch tensor containing pixel data for original image
        - adversary: torch tensor containing pixel data for adversarial image
        - outputs: the output from a forward pass. Should be of shape (n,k),
                   where n = ensemble_size, and k = number of classes
        '''

        gs = gridspec.GridSpec(1,4, width_ratios=[1,1,4,4], wspace=0.6)
        plt.figure(figsize=(16, 4))

        plt.subplot(gs[0])
        if self.dataset == 'cifar':
            plt.imshow(original.squeeze(0).numpy().transpose(1,2,0))
        if self.dataset == 'mnist':
            plt.imshow(original.squeeze(0).squeeze(0).numpy(), cmap='binary_r')
        plt.title('Original Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(gs[1])
        if self.dataset == 'cifar':
            plt.imshow(adversary.squeeze(0).numpy().transpose(1,2,0))
        if self.dataset == 'mnist':
            plt.imshow(adversary.squeeze(0).squeeze(0).numpy(), cmap='binary_r')
        plt.title('Adversarial Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(gs[2])
        colors = ['lightgrey']*10
        colors[original_label] = '#1f77b4' # blue
        plt.bar(range(10), original_output, color=colors)
        plt.xticks(range(10))
        plt.xlabel('Predicted Class')
        plt.ylabel('Fraction of Predictions')
        plt.title('Original Image')
        plt.grid(alpha=0.25)

        plt.subplot(gs[3])
        colors = ['lightgrey']*10
        colors[original_label] = '#1f77b4' # blue
        colors[target_label] = '#ff7f0e' # orange
        plt.bar(range(10), adv_output, color=colors)
        plt.xticks(range(10))
        plt.xlabel('Predicted Class')
        plt.ylabel('Fraction of Predictions')
        plt.title('Adversarial Image')
        plt.grid(alpha=0.25)

        plt.show()



    def make_dataset(self):
        '''
        Use this to make a dataset that you'll use to train a classifier
        for the "indirect approach"
        '''
        pass







