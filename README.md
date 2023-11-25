# Finetune with ResNet50
The primary purpose of this repository is fine-tune `ResNet50` model with `CIFAR10` dataset.

## Initializing

We need to setting up the necessary libraries which include:
- numpy
- torch
- torchvision
- matplotlib
- opencv
- PIL

## Setting up dataset

Firstly, I have get weights of ResNet50 model to extract the transforms for preprocessing dataset. After downloaded the data, I put it into `DataLoader` which is designed to handle large datasets and perform data shuffling, batching, multiprocessing, and merging.

## Modifying model architecture

The architecture of ResNet:

![ResNet-architecture](ResNet-50-architecture.png)  

In this project, I have used CIFAR10 which includes 10 labels.

To finetune the model with custom dataset, we need modify the final layer (fc layer in this model) which has the output units different from our dataset.

## Training model

I have used `CrossEntropyLoss` for criterion and `Adam` with learning rate is `0.001` for optimizer. Due to the limitation of device, I just trained it with 2 epochs.

## Testing with custom data

I used OpenCV to read image. To predict this image, we need to preprocess it using transformer of resnet weights. We need to convert the image to PIL format to meet transformer's requirement.