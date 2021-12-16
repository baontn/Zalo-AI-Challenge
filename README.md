# Zalo-AI-Challenge
This repository contains the code and dataset for the Bremen Big data challenge 2019. The code includes the model using Transfer learning with Inception, an util file, and a model that I built by myself.  
Link to the competition: https://challenge.zalo.ai/portal/5k-compliance  

## Problem description: 

During the Covid-19 outbreak, the Vietnamese government pushed the "5K" public health safety message. In the message, masking and keeping a safe distance are two key rules that have been shown to be extremely successful in preventing people from contracting or spreading the virus. Enforcing these principles on a large scale is where technology may help.

In this challenge, you will create algorithm to detect whether or not a person or group of individuals in a picture adhere to the "mask" and "distance" standards.  
Sample image can be seen below:  
![image](https://user-images.githubusercontent.com/68081679/146218442-53338413-c022-4d2a-8e8e-e9951780514e.png)  

## Data Description:  

### Data Train:  

Inside the train.zip, you can find the following:  

* images: folder stores image file  

* train_meta.csv file: includes 5 columns:  

   image_id: id of image  
   fname: filename of image  
   mask: mask label  
** distancing: distance label  
** 5k: meet both conditions of mask and distancing  
 
Please note that there are some missing labels in the training dataset.   

### Data Public Test:  

Inside the public_test.zip, you can find the following:  

* images: folder stores list image files (jpg)  

* public_test_meta.csv file: includes 2 columns:  

image_id: id of image  
fname: filename of image  

In the private test you can find the same thing in the Private Test file.   

## Data Inspection: 

#### Describe the dataset:  

![image](https://user-images.githubusercontent.com/68081679/146222605-5db362e7-f86f-436c-94b8-fab25c2c7bf0.png)  

It can be observed that there are 4175 images in the train dataset, there were a lot of missing labels. The number of missing labels were:  
![image](https://user-images.githubusercontent.com/68081679/146226304-e4e0286b-d52c-42f6-b30b-8e7a23d9b8f4.png)  
I will only use fully labeled data for my models, and compensate the lack of data by data augmentation.  

## Build the models 

### Transfer Learning using Inception V1 Classification  

![image](https://user-images.githubusercontent.com/68081679/146235168-f09c9b97-be42-4b57-a400-81f762bb0250.png)  
The pretrained model used in this model is Inception V1 Classification. Link to the model from tensorhub: https://tfhub.dev/google/imagenet/inception_v1/classification/5  
The function "create_model" were defined as below:  

















