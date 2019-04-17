#!/usr/bin/env python3

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

def gpu(item):
    return item.cuda() if torch.cuda.is_available() else item

def cpu(item):
    return item.cpu()

def load_data(batch_size, data_transform, dataType='training'):
    # create the transformed dataset
    dataset = FacialKeypointsDataset(csv_file='./data/' + dataType + '_frames_keypoints.csv',
                                           root_dir='./data/' + dataType + '/',
                                           transform=data_transform)

    print('Number of train images:', len(dataset))

    # load training data in batches
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4)

    return (dataset, loader)

# test the model on a batch of test images
def net_sample_output(net, test_loader):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = gpu(images.type(torch.FloatTensor))
        print('batch size', images.shape)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

def train_net(net, n_epochs, loader, criterion, optimizer):
    net.train()

    batch_loss = []
    epoch_loss = []
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
            key_pts = gpu(key_pts.type(torch.FloatTensor))
            images = gpu(images.type(torch.FloatTensor))

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
            batch_loss.append(loss.item())
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0
        epoch_loss.append(running_loss)

    net.eval()
    print('Finished Training')
    return (batch_loss, epoch_loss)


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

def save_model(net):
    ## TODO: change the name to something uniqe for each new model
    import time
    model_dir = 'saved_models/'
    model_name = 'keypoints_model_{}.pt'.format(time.time())

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)

def validate(net, images):
    return net(cpu(images))

if __name__ == '__main__':
    torch.cuda.empty_cache()

    net = gpu(Net())
    print('Defined net:', net)

    data_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        Normalize(),
        ToTensor(),
    ])

    (train_dataset, train_loader) = load_data(batch_size=10, data_transform=data_transform, dataType='training')
    (test_dataset, test_loader) = load_data(batch_size=10, data_transform=data_transform, dataType='test')

    # returns: test images, test predicted keypoints, test ground truth keypoints
    (test_images, test_outputs, gt_pts) = net_sample_output(net, test_loader)

    # print out the dimensions of the data to see if they make sense
    print('test data input size     :', test_images.data.size())
    print('test data output size    :', test_outputs.data.size())
    print('key points size          :', gt_pts.size())

    learning_rate = 1e-4
    criterion = gpu(nn.MSELoss())

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    # train your network
    n_epochs = 1 # start small, and increase when you've decided on your model structure and hyperparams
    batch_loss, epoch_loss = train_net(net, n_epochs, train_loader, criterion, optimizer)

    net = cpu(net)
    test_images = cpu(test_images)
    predicted_key_pts = validate(net, test_images)
    predicted_key_pts = predicted_key_pts.view(predicted_key_pts.size(0), 68, 2)

    visualize_output(test_images, predicted_key_pts, gt_pts)
