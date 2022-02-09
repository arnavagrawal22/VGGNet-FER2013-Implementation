#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
#this statement is used to import print from higher python version to this.
#though we are using python 3, we don't need this, but we are keeping it
#for backwards compatibility.
from __future__ import print_function
#to read and play with data
import numpy as np
import pandas as pd
#to interact with system files and operating system
import os
#to play with images, PIL is python image library
from PIL import Image
#the below library is used to create progress bars in python
from tqdm import tqdm

class GenerateData():
    def __init__(self,datapath):
        self.datapath = datapath
        #this data path is of test and train
        #we are using fer2013 dataset
    
    #this can be done manually too
    def split(self,testfile = "final_test",valfile = "val"):
        #so fer 2013 test data csv has both val and test files mixed. We need to split them.
        #file which needs to be splitted:
        csv_path = self.data_path +"/"+ 'test.csv' 
        #reading using pandas:
        test = pd.read_csv(csv_path)
        #splitting into val:
        validation_data = pd.DataFrame(test.iloc[:3589,:]) #using iloc function of pd
        #test data:
        test_data = pd.DataFrame(test.iloc[3589:,:])

        #saving as test and train:
        test_data.to_csv(self.data_path+"/"+testfile+".csv")
        validation_data.to_csv(self.data_path+"/"+valfile+".csv")
        print("Done splitting the test file into validation & final test file")
    
    def string_to_PIL(self,img_string=" "): #defalt is just an emptystring
        #this function will return a PIL Image object
        pixel_array_string = img_string.split(' ') #this will return a list of substrings
        pixel_array = np.asarray(pixel_array_string,dtype=np.uint8).reshape(48,48) #as fer2013 is 48*48
        return Image.fromarray(pixel_array)
    
    def saving_images(self,dataset_type = "train"): #default is train, but we will do all train/test etc
        #this will create a folder called TRAIN/FINAL_TEST/VAL and save images in it
        foldername = self.data_path+"/"+dataset_type
        dataset_csv_file = self.data_path+"/"+dataset_type+".csv"
        #the below statement makes a file titled above, if it doesn't exist already.
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        csv_data = pd.read_csv(dataset_csv_file)

        image_pixels = csv_data["pixels"] #selecting the pixel column which has space sepearted pixels
        
        no_of_images = image_pixels.shape[0]

        for index in tqdm(range(no_of_images)): #here tqdm is just for progress bar

            img = self.string_to_PIL(image_pixels[index])
            # this is making the path to saving images, nothing fancy
            img.save(os.path.join(foldername,'{}{}.jpg'.format(dataset_type,index)),'JPEG')
        print('Done saving {} data'.format((foldername)))