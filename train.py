#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import time

# watch for any changes in model.py, if it changes, re-load it automatically
#  %load_ext autoreload
#  %autoreload 2

from torch.utils.data import DataLoader
from torchvision import transforms

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor

import torch
import torch.nn as nn
import torch.optim as optim

from models import Net

class Trainer(object):

    def __init__(self, batch_size=10, n_epochs=1, learning_rate=1e-4, useGpu=True):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.useGpu = useGpu
        self.net = self.gpu(Net())
        self.criterion = self.gpu(nn.MSELoss())
        self.learning_rate = 1e-4
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        self.data_transform = transforms.Compose([
            Rescale(250),
            RandomCrop(224),
            Normalize(),
            ToTensor(),
        ])

    def gpu(self, item):
        if self.useGpu and torch.cuda.is_available():
            return item.cuda()
        else:
            return item

    def load_data(self, dataType='training'):
        # create the transformed dataset
        assert(self.data_transform is not None)
        dataset = FacialKeypointsDataset(csv_file='./data/' + dataType + '_frames_keypoints.csv',
                                         root_dir='./data/' + dataType + '/',
                                         transform=self.data_transform)

        print('Number of {0:10} images: {1:6}'.format(dataType, len(dataset)))

        # load training data in batches
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4)

        return (dataset, loader)

    def reload_all_data(self):
        self.train_dataset, self.train_loader = self.load_data(dataType='training')
        self.test_dataset, self.test_loader = self.load_data(dataType='test')

    # test the model on a batch of test images
    def net_sample_output(self):
        # iterate through the test dataset
        assert(self.test_loader is not None)
        for i, sample in enumerate(self.test_loader):

            # get sample data: images and ground truth keypoints
            images = sample['image']
            key_pts = sample['keypoints']

            # convert images to FloatTensors
            images = images.type(torch.FloatTensor)
            print('batch size', images.shape)

            # forward pass to get net output
            output_pts = self.net(images)

            # reshape to batch_size x 68 x 2 pts
            output_pts = output_pts.view(output_pts.size()[0], 68, -1)

            # break after first image is tested
            if i == 0:
                return images, output_pts, key_pts

    def train_net(self):
        assert(self.net is not None)
        assert(self.train_loader is not None)
        assert(self.criterion is not None)
        assert(self.optimizer is not None)

        self.net.train()

        total_batch_loss = []
        epoch_loss = []
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            batch_loss = []
            epoch_start_time = time.time()

            # train on batches of data, assumes you already have loader
            for batch_i, data in enumerate(self.train_loader):
                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['keypoints']

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = self.gpu(key_pts.type(torch.FloatTensor))
                images = self.gpu(images.type(torch.FloatTensor))

                # forward pass to get outputs
                output_pts = self.net(images)

                # calculate the loss between predicted and target keypoints
                # print(output_pts, key_pts)
                loss = self.criterion(output_pts, key_pts)
                #  print(output_pts)
                #  print(key_pts)

                # zero the parameter (weight) gradients
                self.optimizer.zero_grad()

                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                self.optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                batch_loss.append(loss.item())

                print('Epoch: {0:3}, Batch: {1:4}, Batch Loss: {2:12.6f}, Elapsed Time: {3:15.4f} s'.format(epoch + 1, batch_i + 1, running_loss / self.batch_size, time.time() - epoch_start_time))
                running_loss = 0.0

            total_batch_loss.append(batch_loss)
            epoch_loss.append(running_loss)

        print('Finished Training')
        return (total_batch_loss, epoch_loss)


    def visualize_output(self, test_images, test_outputs, gt_pts=None):
        plt.figure(figsize=(40,30))

        self.net.to('cpu')
        self.net.eval()
        for i in range(self.batch_size):
            ax = plt.subplot(self.batch_size / 10, 10, i + 1)

            # un-transform the image data
            image = test_images[i].data   # get the image from it's Variable wrapper
            image = image.numpy()   # convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

            # un-transform the predicted key_pts data
            p_key_pts = test_outputs[i].data
            p_key_pts = p_key_pts.numpy()
            # undo normalization of keypoints
            p_key_pts = p_key_pts * 50.0 + 100

            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[i]
                ground_truth_pts = ground_truth_pts * 50.0 + 100

            # call show_all_keypoints
            self.show_all_keypoints(ax, np.squeeze(image), p_key_pts, ground_truth_pts)

            plt.axis('off')

        plt.show()

    def show_all_keypoints(self, ax, image, predicted_key_pts, gt_pts=None):
        """Show image with predicted keypoints"""
        # image is grayscale
        ax.imshow(image, cmap='gray')
        ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=10, marker='.', c='m')
        # plot ground truth points as green pts
        if gt_pts is not None:
            ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=10, marker='.', c='g')

    def save_model(self):
        model_dir = 'saved_models/'
        model_name = 'keypoints_model_{}.pt'.format(time.time())

        # after training, save your model parameters in the dir 'saved_models'
        torch.save(self.net.state_dict(), model_dir+model_name)

    def validate(self):
        self.net.eval()
        images, predicted, gt = self.net_sample_output(self.net, self.test_loader)
        predicted = predicted.view(predicted.size(0), 68, -1)
        self.visualize_output(images, predicted, gt)

    def plot_batch_loss(batch_loss):
        for loss in batch_loss:
            plt.plot(loss)
        plt.show()

    def train(self):
        print('Start training ...')
        print('Number of epochs {}, batch size {}'.format(self.n_epochs, self.batch_size))
        torch.cuda.empty_cache()

        self.reload_all_data()

        # train your network
        total_batch_loss, epoch_loss = self.train_net()
        self.plot_batch_loss(total_batch_loss)
        # returns: test images, test predicted keypoints, test ground truth keypoints
        (test_images, test_outputs, gt_pts) = self.net_sample_output(self.net, self.test_loader)

        # print out the dimensions of the data to see if they make sense
        print('test data input size     :', test_images.data.size())
        print('test data output size    :', test_outputs.data.size())
        print('key points size          :', gt_pts.size())

        test_images = test_images.cpu()
        predicted_key_pts = self.validate(self.net, test_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CNN training')
    parser.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=10, help='batch size')
    parser.add_argument('--n_epochs', action='store', type=int, dest='n_epochs', default=10, help='number of epochs')
    parser.add_argument('--useGpu', action='store_true', dest='useGpu', help='try to use gpu (cuda)')
    args = parser.parse_args()

    trainer = Trainer(batch_size=args.batch_size, n_epochs=args.n_epochs, useGpu=args.useGpu)
    trainer.train()

