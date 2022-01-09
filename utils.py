# %% [code] {"jupyter":{"outputs_hidden":false}}
import math
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import tensorflow_hub as hub
from tensorflow.keras import layers
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)
from keras import backend as K

def fill_nan(meta): #replace missing 5K label by 0 and 1
    for i in range(len(meta)):
      if math.isnan(meta['mask'][i]) == True and math.isnan(meta['5k'][i]) == True and meta['distancing'][i] == 0:
        meta['5k'][i] = 0
      elif math.isnan(meta['distancing'][i]) == True and math.isnan(meta['5k'][i]) == True and meta['mask'][i] == 0:
        meta['5k'][i] = 0
        return meta

def get_xy(meta):
    '''
    Get the X and Y train/test datasets
    '''

    df = meta.copy()
    a = []
    b=[]
    for i in range(len(df)):
      if math.isnan(df['mask'][i]) == False and math.isnan(df['distancing'][i]) == False:
        a.append(df['fname'][i])
        b.append(i)
    y = []
    for i in range(len(a)):
      j = b[i]
      y.append(df.loc[j, ['mask','distancing']])
    y = np.asarray(y).astype(np.float32)
    target_dir = 'Zalo-AI-Challenge/train-zaloAI/images/'
    SIZE = 360
    x_dataset = []
    for i in range(len(a)):
        img = image.load_img(target_dir +a[i], target_size=(SIZE,SIZE,3))
        img = image.img_to_array(img)
        img = img/255.
        x_dataset.append(img)
    X = np.array(x_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.05)
    return X_train, X_test, y_train, y_test


def get_Xpred(test_dir, csv='private'): #change to public test if needed. Default = private
    csv_path = test_dir + csv + '_test/' + csv + '_test_meta.csv'
    test_dir = test_dir + csv+ '_test/'+ 'images/'
    pub_test = pd.read_csv(csv_path)
    a = pub_test['fname']
    SIZE = 360
    x_pred = []
    for i in range(len(a)):
        img = image.load_img(test_dir +a[i], target_size=(SIZE,SIZE,3))
        img = image.img_to_array(img)
        img = img/255.
        x_pred.append(img)
    x_pred = np.array(x_pred)
    return x_pred, pub_test


def get_predict(x_pred, model):
    '''
    Get the predicted result
    :param x_pred:
    :param model:
    :return:
    '''
    y_pred = model.predict(x_pred)#.argmax(axis=1)
    for i in range(len(y_pred)):
        for j in range(2):
            y_pred[i][j] = round(y_pred[i][j])
    return y_pred


def plot_accloss(r):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(15,15))
    fig.suptitle('Plot the Loss and Accuracy and F1 Score')
    ax1.plot(r.history['accuracy'], label='acc')
    ax1.plot(r.history['val_accuracy'], label='val_acc')
    ax1.legend()
    ax2.plot(r.history['loss'], label='loss')
    ax2.plot(r.history['val_loss'], label='val_loss')
    ax2.legend()
    ax3.plot(r.history['f1_m'], label='f1')
    ax3.plot(r.history['val_f1_m'], label='val_f1')
    ax3.legend()


def built_model():
    '''
    This model create the custom CNN model.
    '''
    from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.models import Model

    i = Input(shape=X_train[0].shape)
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(i, x)
    return model

def create_model(model_url, num_classes=2):

  '''
  Create model from pre-trained model from tensorflow hub

  model_url: url of the pre-trained model
  num_class: number of outcomes, in this case 2

  '''
  size = 360
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=True,
                                           name='feature_extraction_layer',
                                           input_shape=(size,size)+(3,)) # define the input image shape
  model = tf.keras.Sequential([
    feature_extractor_layer, # use the feature extraction layer as the base
    layers.Dense(num_classes, activation='sigmoid', name='output_layer') # create our own output layer
  ])
  return model


def create_result(x_pred, model, pub_test, name='solution'):
    '''
    Create a CSV file of the predicted result
    :param x_pred: The input used to predict
    :param model: The model
    :param pub_test: output
    '''
    name = name + 'csv'
    y_pred= get_predict(x_pred, model)
    y_mask = []
    y_distance = []
    for i in range(len(y_pred)):
        y_mask.append(y_pred[i][0])
        y_distance.append(y_pred[i][1])
    namk = []
    for i in range(len(y_mask)):
        if y_mask[i] == 1 and y_distance[i] ==1:
            namk.append(1)
        else:
            namk.append(0)
    pub_test['5K'] = namk
    pub_test.to_csv(name, index=False)


def tensorboard_hub(dir_name, model_name):
    '''
    Save the metrics for visualizaton
    dir_name: name of the saving directory
    model_name: name of the model
    '''

    log_dir = dir_name + "/" + model_name
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving model to {model_name}")
    return tensor_board


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def compare_hist(r, r1, initial_epoch):
    '''
    Create a graph of before and after data augmentation
    :param r: model before data augmentation
    :param r1: model after data augmentation
    :param initial_epoch: epoch that begins the data augmentation process
    '''
    acc = r.history['accuracy']
    val_acc = r.history['val_accuracy']

    loss = r.history['loss']
    val_loss = r.history['val_loss']

    f1 = r.history['f1_m']
    val_f1 = r.history['val_f1_m']

    new_acc = acc + r1.history['accuracy']
    new_loss = loss + r1.history['loss']

    new_valacc = val_acc + r1.history['val_accuracy']
    new_valloss = val_loss + r1.history['val_loss']

    new_f1 = f1 + r1.history['f1_m']
    new_valf1 = val_f1 + r1.history['val_f1_m']

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(15,15))
    fig.suptitle('Plot the Loss and Accuracy and F1 Score')
    ax1.plot(new_acc, label='acc')
    ax1.plot(new_valacc, label='val_acc')
    ax1.plot([initial_epoch, initial_epoch], plt.ylim(), label = 'fine tune')
    ax1.legend()
    ax2.plot(new_loss, label='loss')
    ax2.plot(new_valloss, label='val_loss')
    ax2.plot([initial_epoch, initial_epoch], plt.ylim(), label = 'fine tune')
    ax2.legend()
    ax3.plot(new_f1, label='f1')
    ax3.plot(new_valf1, label='val_f1')
    ax3.plot([initial_epoch, initial_epoch], plt.ylim(), label = 'fine tune')
    ax3.legend()
