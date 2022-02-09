from __future__ import print_function
from PIL import Image

import numpy  as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# from data_loader import Plain_Dataset, eval_data_dataloader 
from model import DeepEmotion
from data_generation import GenerateData

#setting up device specific
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyperparameters
epochs = 500
lr = 0.005
batch_size = 128

#the following lines of code generate images from data, need to be used only once
# data_gen = GenerateData("data")
# data_gen.split()
# data_gen.saving_images("train")
# data_gen.saving_images("final_test")
# data_gen.saving_images("val")









