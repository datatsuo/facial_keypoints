## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        
        ### My implementation
        # memo: input image 224x224 grayscalye
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 6, stride = 2, padding = 0)
        # size: 110x110x8
        self.conv2 = nn.Conv2d(8, 16, 4, 2, 0)
        # size: 54x54x16
        self.conv3 = nn.Conv2d(16, 32, 4, 2, 0)
        # size: 26x26x32
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 0)
        # size: 12x12x64

        self.fc1 = nn.Linear(12*12*64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2d = nn.Dropout2d(p=0.3)
        self.lrelu = nn.LeakyReLU(0.2)

        # # Xavier uniform initialization (comment out if one wants to use this)
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.conv3.weight)
        # nn.init.xavier_uniform_(self.conv4.weight)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        ### My implementation
        # convolutional layers
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.dropout2d(x)

        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.dropout2d(x)

        x = self.conv4(x)
        x = self.lrelu(x)

        x = x.view(x.size(0), -1) # flattening
        # fully connected layers
        x = self.dropout(x)
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.lrelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
