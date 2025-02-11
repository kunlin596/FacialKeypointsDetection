import torch
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Define all the layers of this CNN, the only requirements are:
        #
        #   1. This network takes in a square (same width and height), grayscale image as input
        #   2. It ends with a linear layer that represents the key points
        #
        # It's suggested that you make this last layer output 136 values,
        # two for each of the 68 keypoint (x, y) pairs.
        # As an example, you've been given a convolutional layer,
        # which you may (but don't have to) change:
        #   1. input image channel (grayscale), 32 output channels/feature maps,
        #      5x5 square convolution kernel

        # For computing the input size
        dummy_data = torch.randn(20, 1, 224, 224)

        self.pool = nn.MaxPool2d(2, 2)  # kernel size and stride size

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.dropout1 = nn.Dropout(p=0.1)
        output_layer1 = self.pool(self.conv1(dummy_data))

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout2 = nn.Dropout(p=0.2)
        output_layer2 = self.pool(self.conv2(output_layer1))

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dropout3 = nn.Dropout(p=0.3)
        output_layer3 = self.pool(self.conv3(output_layer2))

        self.conv4 = nn.Conv2d(128, 256, 2)
        self.dropout4 = nn.Dropout(p=0.4)
        output_layer4 = self.pool(self.conv4(output_layer3))

        output_layer4 = output_layer4.view(output_layer4.shape[0], -1)

        self.fc1 = nn.Linear(output_layer4.shape[1], 1000)
        #  self.fc1.weight.data.normal_(std=0.02)

        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)
        #  self.fc2.weight.data.normal_(std=0.02)

        self.dropout6 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(1000, 68 * 2)

        #  self.fc3.weight.data.normal_(std=0.02)
        ## Note that among the layers to add, consider including:
        # max pooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid over fitting

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
