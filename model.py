import torch.nn.functional as F
import torch
from torch import nn


class Net(nn.Module):

  '''
  Define all the layers of this CNN, the only requirements are:
  1. This network takes in a square (same width and height), grayscale image as input
  2. It ends with a linear layer that represents the keypoints
  Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
  '''

  def __init__(self):
    super().__init__()

    # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
    
    ## Shape of a Convolutional Layer
    # K - out_channels : the number of filters in the convolutional layer
    # F - kernel_size
    # S - the stride of the convolution
    # P - the padding
    # W - the width/height (square) of the previous layer
    
    # Since there are F*F*D weights per filter
    # The total number of weights in the convolutional layer is K*F*F*D
    
    # 224 by 224 pixels
    
    ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    

    # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
    # output tensor after conv1 (32, 220, 220,1)
    self.conv1 = nn.Conv2d(1,32,5)

    # maxpool layer
    # pool with kernel_size=2, stride=2
    self.pool = nn.MaxPool2d(2,2)

    # 220/2 = 110
    # after max pool1 : (32, 110, 110,1)
    # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
    # output tensor after conv2 (64, 108, 108,1)
    self.conv2 = nn.Conv2d(32,64,3)

    # After max pool from conv2 layer tensor shape = (64,54,54,1)
    # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
    # output tensor after conv3 (128, 52, 52,1)
    self.conv3 = nn.Conv2d(64,128,3)

    # After max pool from conv3 layer tensor shape = (128,26,26,1)
    # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
    # output tensor after conv4 (256, 24, 24,1)
    self.conv4 = nn.Conv2d(128, 256, 3)

    # After max pool from conv4 layer tensor shape = (256,12,12,1)
    # output size = (W-F)/S +1 = (12-3)/1 + 1 = 12
    # output tensor after conv1 (512, 12, 12,1)
    self.conv5 = nn.Conv2d(256, 512, 1)

    # After max pool from conv5 layer tensor shape = (512,6,6,1)
    # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
    # output tensor(512,6,6,1)

    # Fully-connected (linear) layers
    self.fc1 = nn.Linear(512*6*6, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 68*2)
    
    # Dropout
    self.dropout = nn.Dropout(p=0.25)

  def forward(self, x):
    ## Define the feedforward behavior of this model
    ## x is the input image and, as an example, here you may choose to include a pool/conv step:
    
    # 5 conv/relu + pool layers
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = self.pool(F.relu(self.conv5(x)))
    
    # Prep for linear layer / Flatten
    x = x.view(x.size(0), -1)
    
    # linear layers with dropout in between
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    
    return x