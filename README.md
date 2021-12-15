# Zalo-AI-Challenge
This repository contains the code and dataset for the Bremen Big data challenge 2019.  
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
distancing: distance label  
5k: meet both conditions of mask and distancing  
 
Please note that there are some missing labels in the training dataset.   

### Data Public Test:  

Inside the public_test.zip, you can find the following:  

* images: folder stores list image files (jpg)  

* public_test_meta.csv file: includes 2 columns:  

image_id: id of image  
fname: filename of image  

