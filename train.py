#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np

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

net = Net()
print('Net:', net)

def load_train_data(data_transform):
    # create the transformed dataset
    train_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                           root_dir='./data/training/',
                                           transform=data_transform)


    print('Number of train images:', len(train_dataset))

    # iterate through the transformed dataset and print some stats about the first few samples
    # for i in range(4):
    #     sample = train_dataset[i]
    #     print('test sample image: ', i, 'image size:', sample['image'].size(), 'key points size:', sample['keypoints'].size())

    # load training data in batches
    batch_size = 10
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    return (train_dataset, train_loader)


def load_test_data(data_transform):
    # create the test dataset
    test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                          root_dir='./data/test/',
                                          transform=data_transform)

    print('Number of test images:', len(test_dataset))

    # load test data in batches
    batch_size = 10

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

    return (test_dataset, test_loader)

# test the model on a batch of test images
def net_sample_output(test_loader):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)
        print('batch size', images.shape)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

def train_net(n_epochs, loader, criterion, optimizer):
    net.to('cuda')
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have loader
        for batch_i, data in enumerate(loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            images = images.to('cuda')
            key_pts = key_pts.to('cuda')
            #  print(key_pts)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            # print(output_pts, key_pts)
            loss = criterion(output_pts, key_pts)
            #  print(output_pts)
            #  print(key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

    net.eval()
    print('Finished Training')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    plt.figure(figsize=(40,30))

    net.to('cpu')
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = (predicted_key_pts * 50.0 + 100)

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(ax, np.squeeze(image), predicted_key_pts, ground_truth_pts)

        # plt.axis('off')

    plt.show()

def show_all_keypoints(ax, image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=10, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=10, marker='.', c='g')

def save_model(name):
    ## TODO: change the name to something uniqe for each new model
    model_dir = 'saved_models/'
    model_name = 'keypoints_model_{}.pt'.format(name)

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)

def validate(images):
    return net(images)
    

if __name__ == '__main__':

    ## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
    # order matters! i.e. rescaling should come before a smaller crop
    #
    # the default transforms operate upon the np.arrays, we need a custom transform object to handle dict.
    # see data_load.py for more details.

    data_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        Normalize(),
        ToTensor(),
    ])

    # testing that you've defined a transform
    assert(data_transform is not None), 'Define a data_transform'

    (train_dataset, train_loader) = load_train_data(data_transform)
    (test_dataset, test_loader) = load_test_data(data_transform)


    # returns: test images, test predicted keypoints, test ground truth keypoints
    test_images, test_outputs, gt_pts = net_sample_output(test_loader)

    # print out the dimensions of the data to see if they make sense
    print('test data input size', test_images.data.size())
    print('test data output size', test_outputs.data.size())
    print('key points size', gt_pts.size())

    learning_rate = 1e-4
    criterion = nn.MSELoss().cuda()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    # train your network
    n_epochs = 10 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs, train_loader, criterion, optimizer)

    predicted_key_pts = validate(test_images)
    visualize_output(test_images, predicted_key_pts, gt_pts)
