#nn.Conv2d(in channel, outchannels, kernel size)
#nn.MaxPool2d(kernel,stride)

import torch
import torch.nn as nn
import torch.functional as F

class DeepEmotion(nn.Module): #inherits nn.module here
    
    def __init__(self):
        super(DeepEmotion, self).__init__()
        # inchannels = 1 (gray scale images), outchannels = 10 (no. of fliters), filter size = 3 * 3 
        self.conv_1 = nn.Conv2d(1,10,3) #48*48
        # 10 from prev, 10 outgoing, 3x3 kernel
        self.conv_2 = nn.Conv2d(10,10,3) #46 * 46
        # kernel size = 2, stride = 2
        self.pool_1 = nn.MaxPool2d(2,2) #23 * 23
        #----------------------------------------------------------------#    
        self.conv_3 = nn.Conv2d(10,10,3) #21 * 21
        self.conv_4 = nn.Conv2d(10,10,3) #19 * 19
        self.pool_2 = nn.MaxPool2d(2,2) #9*9
        #----------------------------------------------------------------#
        self.norm = nn.BatchNorm2d(10) #num_features – C from an expected input of size (N, C, H, W)
        #image size just before this is 9*9 with 10 channels
        self.fc1 = nn.Linear(810,50) #50 hidden layer according to Deep Emotion paper
        self.fc2 = nn.Linear(50,7) #there are 7 emotions
        #----------------------------------------------------------------#
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, 7), # 42 * 42
            nn.MaxPool2d(2, 2), # 21 * 21
            nn.ReLU(True), #True, because we need this to happen implace and we will not call it again
            nn.Conv2d(8, 10, 5), #17 * 17
            nn.MaxPool2d(2, 2),# 8 * 8
            nn.ReLU(True)
        ) 
        self.fc_loc = nn.Sequential(
            #input to this fully connected layer will be a flattened 640 pixel image
            nn.Linear(640, 32),
            #according to Deep Emotion paper 32
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            #affine grid, which is grid generator in pytorch requires a 2*3 input grid for 2D images
        )
        #when we say fc_loc[2] we are refferring to the Last nn.linear layer, its basically like a list
        self.fc_loc[2].weight.data.zero_()
        #.weight is parameter of that later, .data is its data, and .zero_ initializes them to zero implace
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype = torch.float32))
        #same as above, just we are initiasing it with an identity matrix, if you flatten it  
        #----------------------------------------------------------------#
        #spatial transformer network : 
        def spatial(self,x):
            temp = self.localization(x)
            #here output will be 10 channel, 8*8 image = 640
            #after flattening
            temp = temp.view(-1,640)
            theta = self.fc_loc(temp)
            theta = theta.view(-1,2,3)
            #we are doing this reshaping as affine grid needs 2*3
            grid_gen = F.affine_grid(theta,x.size()) #x.size() as this is the target output image size,N*48*48*1            
            x = F.grid_sample(x,grid_gen) #x is input, and it takes the grid to map it to x
            return x
        #----------------------------------------------------------------#
        def forward(self,input):
            #----------------------------------------------------------------#
            out = self.spatial(input)
            #----------------------------------------------------------------#
            out = F.ReLU(self.conv_1(out))
            out = self.conv_2(out)
            out = F.ReLU(self.pool_1(out))
            #----------------------------------------------------------------#
            out = F.ReLU(self.conv_3(out))
            out = self.conv_4(out)
            out = F.ReLU(self.pool_2(out))
            #----------------------------------------------------------------#
            out = F.dropout(out)
            # 9*9 image and *10 channels
            out = out.view(-1,810)
            out = F.ReLU(self.fc_1(out))
            out = self.fc_2(out)
            #----------------------------------------------------------------#
            return out