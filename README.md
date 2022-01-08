# Zalo-AI-Challenge
This repository contains the code and dataset for the Zalo AI challenge 2021. It includes the model using Transfer learning with multiple pre-trained models, an util file, and a model that I built by myself.  
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
I will only use fully labeled data for my models and compensate for the lack of data by data augmentation.  

## Build the models 

### Transfer Learning using pre-trained models  

![image](https://user-images.githubusercontent.com/68081679/147751374-4e7f262f-29f2-4e8d-90e3-cd81738f1f14.png)  
The links above are the links to the pre-trained models from TensorFlow Hub.    
The function "create_model" were defined as below:  

![image](https://user-images.githubusercontent.com/68081679/147783920-6869afb0-0f38-4998-8227-ef349873cca1.png)  
The model will be created from the pre-trained models from the Tensorflow hub. I choose [Imagenet (ILSVRC-2012-CLS) classification with Inception V3](https://tfhub.dev/google/imagenet/inception_v3/classification/5), [Feature vectors of images with EfficientNet V2 with input size 224x224, trained on imagenet-ilsvrc-2012-cls](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2), [Feature vectors of images with EfficientNet V2 with input size 480x480, trained on imagenet-21k (Full ImageNet, Fall 2011 release)](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2).  

Initially, the trainable attribute was set to False, but then I turned to True and it produced much better performance. I used SGD optimizer, batch size of 32, and let the model run for 10 epochs. After that, I used Data Augmentation to improve the model. The Image size was set to be 360x360. However, for the model EfficienNet V2 trained on imagenet 21k, I only trained for 4 epochs due to its long training time with the batch-size of 4. 
![image](https://user-images.githubusercontent.com/68081679/147784591-c94fadc4-9a1f-4c02-b45d-4ace45da3154.png)  

The models were saved into h5 files. 

### Self-built model using CNN layers.
![image](https://user-images.githubusercontent.com/68081679/147785354-ddca6c5a-bb62-44b2-bd7c-88e759df7f53.png)  
In the self-built model, I used multiple Conv2D layers, together with BatchNormalization and MaxPooling2D layers to reduce the dimension of the tensors. I trained the model for 15 epochs with a batch size of 32 before doing Data Augmentation. 

### Image Augmentation

![image](https://user-images.githubusercontent.com/68081679/147786155-b7486cee-56b4-467b-99b2-3ea9caf0e9ec.png)  
Since we had a limited number of fully labeled images, data augmentation was a must-have tool to improve the model. For that, I used the ImageDataGenerator and retrain the models for 10 epochs (for EfficienNet V2 trained on imagenet 21k I still trained for 4 epochs). 

## Result

### Evaluation metrics

The evaluation method for this competition is F1 score. In statistical analysis of binary classification, the F-score or F-measure is a measure of a test's accuracy. It is calculated from the precision and recall of the test, where the precision is the number of **true positive** results divided by the number of **all positive results, including those not identified correctly**, and the recall is the number of **true positive** results divided by the number of **all samples that should have been identified as positive**. Precision is also known as positive predictive value, and recall is also known as sensitivity in diagnostic binary classification. ([Wikipedia, 2021](https://en.wikipedia.org/wiki/F-score)). Beside F1 score, I also used Accuracy.   

### Using Tensorboard to display experiment results  

#### Before Data Augmentation: 

I used Tensorboard to summary the experiment results. [This is the link to the experiment](https://tensorboard.dev/experiment/EqAm4Y6SRAKZBaFZDK8FnA/#scalars&runSelectionState=eyJjdXN0b21fY25uIGF1Z21lbnRhdGlvbi90cmFpbiI6ZmFsc2UsImN1c3RvbV9jbm4gYXVnbWVudGF0aW9uL3ZhbGlkYXRpb24iOnRydWUsImVmZmljaWVudG5ldF92Ml9pbWduZXQxa19iMCBhdWdtZW50YXRpb24vdHJhaW4iOmZhbHNlLCJlZmZpY2llbnRuZXRfdjJfaW1nbmV0MWtfYjAgYXVnbWVudGF0aW9uL3ZhbGlkYXRpb24iOnRydWUsImVmZmljaWVudG5ldF92Ml9pbWduZXQyMWtfYjMgYXVnbWVudGF0aW9uL3RyYWluIjpmYWxzZSwiZWZmaWNpZW50bmV0X3YyX2ltZ25ldDIxa19iMyBhdWdtZW50YXRpb24vdmFsaWRhdGlvbiI6dHJ1ZSwiaW5jZXB0aW9udjMvdHJhaW4iOmZhbHNlLCJpbmNlcHRpb252MyBhdWdtZW50YXRpb24vdHJhaW4iOmZhbHNlLCJlZmZpY2llbnRuZXRfdjJfaW1nbmV0MjFrX2IzL3RyYWluIjpmYWxzZSwiZWZmaWNpZW50bmV0X3YyX2ltZ25ldDFrX2IwL3RyYWluIjpmYWxzZSwiY3VzdG9tX2Nubi90cmFpbiI6ZmFsc2UsImluY2VwdGlvbnYzIGF1Z21lbnRhdGlvbi92YWxpZGF0aW9uIjp0cnVlLCJjdXN0b21fY25uL3ZhbGlkYXRpb24iOmZhbHNlLCJlZmZpY2llbnRuZXRfdjJfaW1nbmV0MWtfYjAvdmFsaWRhdGlvbiI6ZmFsc2UsImVmZmljaWVudG5ldF92Ml9pbWduZXQyMWtfYjMvdmFsaWRhdGlvbiI6ZmFsc2UsImluY2VwdGlvbnYzL3ZhbGlkYXRpb24iOmZhbHNlfQ%3D%3D&_smoothingWeight=0). Below is the graph which demonstrates the F1 scores of my models before data augmentation:  
![image](https://user-images.githubusercontent.com/68081679/148650479-942a31c1-a6ff-4a00-8a65-22248b0ece6d.png)  
The Efficientnet V2 imgnet21k performed best eventhough it had the least number of epochs. The other models (custom CNN, Efficientnet V2 imgnet1k and InceptionV3) seems to stop improving after the 4th epochs and the F1 score even dropped after the 6th epoch. At the end of the training process, the validation F1 score of the Efficientnet V2 imgnet 21k is 0.86, that of Efficientnet V2 imgnet 1k, Inception V3 and Custom CNN were 0.79, 0.76 and 0.74 respectively.  
![image](https://user-images.githubusercontent.com/68081679/148651828-b5817547-4e8b-4a2b-a60e-4776cf0d1f37.png)  
With regard to Loss, the EfficientNet V2 imgnet 21k was the only model that the loss improved during the training time. Other models tended to increase or fluctuate. The Inception V3 model was the with the highest loss and didn't improve through out the training period.  

#### After Data Augmentation: 

The models after data augmentation didn't perform as good as expected, and Efficientnet V2 imgnet 21k remained the best model. The F1 score for the models after data augmentation can be seen below:  
![image](https://user-images.githubusercontent.com/68081679/148652440-f1b6832e-c158-4015-b400-fa63a1f22dd7.png)  
The Efficientnet V2 imgnet 1k, Inception V3 and Efficientnet V2 imgnet 21k slightly increase with the score of 0.83, 0.79 and 0.88 respectively. However, the learning curves were flat. In contrast, that of the Custom CNN model fluctuated and end up worse than before data augmentation with the F1 score of 0.048.   

![image](https://user-images.githubusercontent.com/68081679/148653585-9f58596d-2186-476d-bc88-035e6c2355ca.png)  


Regarding Loss, that of Efficientnet V2 imgnet21k and Inception V3 slightly increased. However that of Efficientnet V2 imgnet 1k and the custom CNN model fluctuated, escpecially the custom CNN model in which it sky-rocketed in the last epoch. 

## Reflection

This is one of my very first model using neural network, and many things are not perfect. There are a lot of rooms for improvement. There are more than 4000 images in the given dataset, but I only used more than 2300 of fully labelled images. I think if I can utilize the remanining images, the models' perfomance can be improved. However I don't know how to approach yet. 






