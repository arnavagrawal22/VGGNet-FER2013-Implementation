from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from data_loader import CustomDataset
from model import VGGNet
from data_generation import GenerateData

# setting up device specific
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
epochs = 50
lr = 0.01
batch_size = 64


# the following lines of code generate images from data, need to be used only once
# data_gen = GenerateData("data")
# data_gen.split()
# data_gen.saving_images("train")
# data_gen.saving_images("test")
# data_gen.saving_images("val")

net = VGGNet()
net.to(device)  # if gpu, model on gpu
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                      nesterov=True, weight_decay=0.0001)

# print("The device used is ", device)
# print("Model architecture: ", net)

# the following lines of code will work only when data has been unpacked
train_csv_file = 'data'+'/'+'train.csv'
validation_csv_file = 'data'+'/'+'val.csv'
train_img_dir = 'data'+'/'+'train/'
validation_img_dir = 'data'+'/' + 'val/'

# 1. Creating a transformation object with Normalizing
transformation = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = CustomDataset(csv_file=train_csv_file,
                              img_dir=train_img_dir,
                              datatype='train',
                              transform=transformation)

validation_dataset = CustomDataset(csv_file=validation_csv_file,
                                   img_dir=validation_img_dir,
                                   datatype='val',
                                   transform=transformation)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(validation_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)


#----------------------------------------------------------------#
# TRAINING FUNCTION
#----------------------------------------------------------------#

def train(epochs, train_loader, val_loader, criterion, optimizer, device):
    print("Training Started...")
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        val_loss = 0
        train_correct = 0
        val_correct = 0

        # setting mode to training, effects Dropout and BatchNorm2d
        net.train()
        for data, labels in train_loader:
            # sending to GPU
            data, labels = data.to(device), labels.to(device)
            # zeroing the gradients
            optimizer.zero_grad()
            # getting output- forward pass
            outputs = net(data)
            # calculating loss
            loss = criterion(outputs, labels)  # labels are real/actual labels
            # calculating gradients
            loss.backward()
            # back propagation
            optimizer.step()

            train_loss += loss.item()  # adding batch loss to the epoch loss

            # getting predictions:
            # will return index of nearest to 1/highest probability
            _, preds = torch.max(outputs, 1)

            train_correct += torch.sum(preds == labels.data)

        # setting the mode to evaluating mode
        net.eval()
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            val_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss = val_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)

        print('Epoch: {} \t Training Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'.format(epoch+1,
                                                                                                                                             train_loss,
                                                                                                                                             validation_loss,
                                                                                                                                             train_acc * 100,
                                                                                                                                             val_acc*100))

    torch.save(net.state_dict(),
               'vgg_net-{}-{}-{}.pt'.format(epochs, batch_size, lr))
    print("Finished Training")

# train(epochs,train_loader,val_loader,criterion,optimizer,device)

# We have trained for 35 epochs


net = VGGNet()
# loading our model we trained on a Kaggle Instance
net.load_state_dict(torch.load('vgg_net-35-64.pt',map_location=torch.device('cpu')))
net.to(device)
print("Model loaded successfully")


test_dataset = CustomDataset(csv_file="data/test.csv",
                             img_dir="data/test/",
                             datatype='test',
                             transform=transformation)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

# val_correct = 0
# print("Calculating Validation Accuracy...")
# for data, labels in tqdm(val_loader):
#             data, labels = data.to(device), labels.to(device)
#             val_outputs = net(data)
#             _, val_preds = torch.max(val_outputs, 1)
#             val_correct += torch.sum(val_preds == labels.data)

# val_acc = val_correct.double() / len(validation_dataset)

# print("Val Accuracy from saved model is ", val_acc.item()*100)

# test_correct = 0
# print("Calculating Test Accuracy...")
# for data, labels in tqdm(test_loader):
#             data, labels = data.to(device), labels.to(device)
#             test_outputs = net(data)
#             _, test_preds = torch.max(test_outputs, 1)
#             test_correct += torch.sum(test_preds == labels.data)

# test_acc = test_correct.double() / len(test_dataset)

# print("Test Accuracy from saved model is ", test_acc.item()*100)


# Testing any image of choice :
#uncomment the following lines to use
# #----ENTER YOUR IMAGE PATH HERE-----#
# frame = cv2.imread("suprised.jpeg")  # reads an image from path
# #-----------------------------------#
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscaling
# # showing grayscaled image:
# faceCascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # detecting faces, 1.1 scale factor, and 4 min neighbors
# faces = faceCascade.detectMultiScale(gray, 1.1, 4)
# # faces is a list of all faces rectangles found
# for x, y, w, h in faces:  # x,y start coordinate, and then width and height
#     # grey region of interest
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = frame[y:y+h, x:x+w]
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0),
#                   2)  # make a rectangle of thickness 2
#     # color blue
#     # detecting face in the region of interest(basically making a new image for face)
#     face_img = faceCascade.detectMultiScale(roi_gray)
#     if len(face_img) == 0:
#         print("Searching others")
#     else:
#         print("Face/s found")
#         for (ex, ey, ew, eh) in face_img:
#             face_roi = roi_color[ey: ey+eh, ex: ex+ew]

# # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# # plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
# # plt.show()
# gray_temp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
# final_image = cv2.resize(gray_temp, (48, 48))
# final_image = np.expand_dims(final_image, axis=0) # adding one dimension in the start
# final_image = np.expand_dims(final_image, axis=0) #adding another dimension in the start
# #the above two are needed, as we need 1 channel and 1 number of image
# final_image = final_image/255.0 #normalizing
# pixel_data = torch.from_numpy(final_image)
# pixel_data = pixel_data.type(torch.FloatTensor)
# outputs = net(pixel_data)
# Pred = F.softmax(outputs, dim=1)
# Predictions = torch.argmax(Pred)
classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
# print(classes[Predictions.item()])

#LIVE WEBCAM DEMO#


# Live Demo
path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set rectangle background to white
rectangle_bgr = (255,255,255)
#make a black image
img = np.zeros((500,500))
#set some text
text = "Some text in a box!"
#get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
#set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
#make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color = (0,0,0), thickness = 1)

cap = cv2.VideoCapture(0)
#check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")

while True:
    ret,frame = cap.read()
    #eye_cascade
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    flag = 0
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")            
        else:
            for (ex,ey,ew,eh) in facess:
                flag = 1;
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]
    if(flag):
        graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        final_image = cv2.resize(graytemp, (48,48))
        final_image = np.expand_dims(final_image, axis = 0)
        final_image = np.expand_dims(final_image, axis = 0)
        final_image = final_image/255.0
        pixel_data = torch.from_numpy(final_image)
        pixel_data = pixel_data.type(torch.FloatTensor)
        pixel_data = pixel_data.to(device)
        outputs = net(pixel_data)
        Pred = F.softmax(outputs, dim = 1)
        Predictions = torch.argmax(Pred)
        print(Predictions)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN
        
        if ((Predictions)==0):
            status = "Angry"            
            
        elif ((Predictions)==1):
            status = "Disgust"
            
        elif ((Predictions)==2):
            status = "Fear"
            
        elif ((Predictions)==3):
            status = "Happy"
            
        elif ((Predictions)==4):
            status = "Sad"
            
        elif ((Predictions)==5):
            status = "Surprise"
            
        elif ((Predictions)==6):
            status = "Neutral"

        x1,y1,w1,h1 = 0,0,175,75
        #black background
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #text
        cv2.putText(frame, status, (x1+int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
        cv2.putText(frame, status, (100,150), font, 3 , (0,0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255))
            
    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
