# Facial Emotion Recognition (VGGNet)

![](https://www.researchgate.net/profile/Martin-Kampel/publication/311573401/figure/fig1/AS:613880195723264@1523371852035/Example-images-from-the-FER2013-dataset-3-illustrating-variabilities-in-illumination.png "FER 2013")

## Brief Description
**Implemented in PyTorch**

This project is a custom implementation of VGGNet trained to classify images of faces into 7 classes, representing different emotions.

The code is highly commented, explaining each step of the process.

## Key Highlights
*We ran only 35 epochs instead of 300, and our 35 epoch checkpoint can be accessed here [35-Epoch-Checkpoint](https://ufile.io/4jsih83r)*

- 62% Test Accuracy against 72% of official implementation
- Live Webcam Demo
- Ability to test custom images

## Acknowledgement

### This is implementation of the paper - 
[Facial Emotion Recognition: State of the Art Performance on FER2013](https://arxiv.org/abs/2105.03588v1)
- [Official Implementation](https://github.com/usef-kh/fer)

## Resources Used for Data Generation/Loading and Live Webcam Demo

- [OmarSayed7's implementation of DeepEmotion2019](https://github.com/omarsayed7/Deep-Emotion)
- [DeepLearning_by_PhDScholar](https://www.youtube.com/watch?v=yN7qfBhfGqs&ab_channel=DeepLearning_by_PhDScholar)



## Dataset

[FER2013](https://www.kaggle.com/deadskull7/fer2013/activity)

## Code Structure

[model.py](./model.py) : This file contains the VGGNet Model with slight modifications

[data_generation.py](./data_generation.py) : This file splits the data into test, train, and val.

[data_loader.py](./data_loader.py) : This is the custom dataset and dataset loader

[main.py](./main.py) : This has the training loop, as well has we can test custom images , and it also has code to do a live webcam test.



## How to Run?

1. #### Preparation of data

- Download the data
- Unpack fer2013.csv
- Place it in ``` pwd/data/ ```

2. Uncomment the data generation lines (First Run only)
3. By default, the model uses pre-trained parameters, to train the model yourself, you can uncomment the train function call
4. By default, ```python main.py ``` will open the webcam demo.

#### Thank You






