#all torchvision transforms are on PIL images, so if we have a csv file or something, then we can change it to PIL image then apply transformers. In our case, we have Images, so we can open those images using Image.open and then return them
#https://debuggercafe.com/custom-dataset-and-dataloader-in-pytorch/ a good reference

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset): #inherits from torch.utils.data.dataset
#we are overriding init, length and getitem
    #datatype is basically dataset type
    def __init__(self,csv_file,img_dir,datatype,transform):
        self.csv_file = pd.read_csv(csv_file)
        self.lables = self.csv_file["emotion"]
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self,idx):
        
        #getitem will be called by dataloader, which can give indexes as a torch list
        if torch.is_tensor(idx):
            idx = idx.tolist() #makes a list of indexes
        #above- if we have a idx torch tensor, convert it to a list
        
        #the below line return an PIL image object
        img = Image.open(self.img_dir+self.datatype+str(idx)+'.jpg')

        #returning the image label
        lables = np.array(self.lables[idx])

        lables = torch.from_numpy(lables).long() #converts to int64

        if self.transform : #if we have mentioned any transformation object, we are calling it here
            img = self.transform(img)
        
        return img,lables #returns pixel values, followed by labels.
