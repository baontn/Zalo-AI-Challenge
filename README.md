# Zalo-AI-Challenge
This repository contains the code and dataset for the Zalo AI challenge 2021. It includes the model using Transfer learning with multiple pre-trained model, an util file, and a model that I built by myself.  
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

    * image_id: id of image  
    * fname: filename of image  
    * mask: mask label  
    * distancing: distance label  
    * 5k: meet both conditions of mask and distancing  
 
Please note that there are some missing labels in the training dataset.   

### Data Public Test:  

Inside the public_test.zip, you can find the following:  

* images: folder stores list image files (jpg)  

* public_test_meta.csv file: includes 2 columns:  

    * image_id: id of image  
    * fname: filename of image  

In the private test you can find the same thing in the Private Test file.   

## Data Inspection: 

#### Describe the dataset:  

![image](https://user-images.githubusercontent.com/68081679/146222605-5db362e7-f86f-436c-94b8-fab25c2c7bf0.png)  

It can be observed that there are 4175 images in the train dataset, there were a lot of missing labels. The number of missing labels were:  
![image](https://user-images.githubusercontent.com/68081679/146226304-e4e0286b-d52c-42f6-b30b-8e7a23d9b8f4.png)  
I will only use fully labeled data for my models, and compensate the lack of data by data augmentation.  

## Build the models 

### Transfer Learning using different models  

![image](https://user-images.githubusercontent.com/68081679/147751374-4e7f262f-29f2-4e8d-90e3-cd81738f1f14.png)  
The links above are the links to the pre-trained models from Tensorflow Hub.    
The function "create_model" were defined as below:  

![image](https://user-images.githubusercontent.com/68081679/147783920-6869afb0-0f38-4998-8227-ef349873cca1.png)
The model will be created from the pre-trained models from Tensorflow hub. I choosed [Imagenet (ILSVRC-2012-CLS) classification with Inception V3](https://tfhub.dev/google/imagenet/inception_v3/classification/5), [Feature vectors of images with EfficientNet V2 with input size 224x224, trained on imagenet-ilsvrc-2012-cls](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2), [Feature vectors of images with EfficientNet V2 with input size 480x480, trained on imagenet-21k (Full ImageNet, Fall 2011 release)](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2).  

Initially, the trainable attribute was set to False, but then I turned to True and it produced much better performance. I let the model run for 10 epochs and use . The Image size was set to be 360x360,

The models were saved into h5 files. 


### Self-built model using CNN layers.





## Result

### Evaluation metrics

The evaluation method for this competition is F1 score. In statistical analysis of binary classification, the F-score or F-measure is a measure of a test's accuracy. It is calculated from the precision and recall of the test, where the precision is the number of **true positive** results divided by the number of **all positive results, including those not identified correctly**, and the recall is the number of **true positive** results divided by the number of **all samples that should have been identified as positive**. Precision is also known as positive predictive value, and recall is also known as sensitivity in diagnostic binary classification. ([Wikipedia, 2021](https://en.wikipedia.org/wiki/F-score)). Beside F1 score, I also use Accuracy. 





















