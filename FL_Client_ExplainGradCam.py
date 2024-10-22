#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import loadtxt
import os
import scipy
import collections
import pathlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import PIL.Image
from datetime import datetime

import tensorflow as tf

import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, multiply

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Activation 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers

from sklearn.model_selection import train_test_split

import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from numpy.random import seed

# Set seeds

seed(1)
tf.random.set_seed(2)
tf.compat.v1.set_random_seed(2019)


# In[2]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is starting =", current_time)
print('***************************** Iteration 1 Starts ***************************** ')


# # Data Covid

# In[3]:


data_dir = pathlib.Path("C:/Users/tapom/Desktop/Research/2_Code/Model with full data/DataCovidHoldOut")
image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[4]:


#Rescaling the image
image_convert = ImageDataGenerator(rescale = 1./255)

#Flow the images from the input directory into an array, applies the pre-processing defined in image_convert function, resizes all images in same size 
#image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[3, 3], class_mode="binary", batch_size=1000,seed=32)
image_preprocess = image_convert.flow_from_directory(directory=data_dir, target_size=[180, 180], class_mode="binary", batch_size=4000,seed=32)
data, labels = image_preprocess.next()


# In[5]:


labels = labels. astype(int)
print(type(data))
print(data.shape)
print(type(labels))
print(labels.shape)
print(sum(labels))

print(labels)


# In[6]:


# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=40)

print(X_train.shape)
print(y_train.shape)
print(sum(y_train))

print(X_test.shape)
print(y_test.shape)
print(sum(y_test))


# In[7]:


def SqueezeAndExcitation(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = multiply([inputs,x])
    return x


# In[8]:


input_img = Input(shape=(180, 180, 3))

x = Conv2D(16, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = SqueezeAndExcitation(x)                                    #Squeeze and Excitation Attention Layer

x = Flatten()(x)

x = Dense(550, activation='relu')(x)
x = Dropout(0.1,seed=2019)(x)

x = Dense(400, activation='relu')(x)
x = Dropout(0.3,seed=2019)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.2,seed=2019)(x)

output = Dense(1, activation='sigmoid')(x)                  # Add Output layer

model = Model(inputs=input_img, outputs=output)


# In[9]:


# Print the model summary
model.summary()


# In[10]:


model.compile(optimizer='Adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model is ignored and Final FL model weights are loaded directly instead
#r = model.fit(X_train,y_train, epochs=2, batch_size=1000, shuffle=True)


# In[11]:


# Read model weight from Central Server
Central_weight = pickle.load(open('Central_Model2_Covid.pkl', 'rb'))

#apply new weights to model
model.set_weights(Central_weight)


# In[12]:


# Print final model performance

print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))


# In[13]:


# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
#yhat_classes = model.predict_classes(X_test, verbose=0)


yhat_classes_1 =model.predict(X_test, verbose=0)

yhat_classes = np.where(yhat_classes_1 > 0.5, 1,0)

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# In[14]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when code is ending =", current_time)
print('***************************** Iteration 1 Ends ***************************** ')


# In[ ]:





# In[ ]:





# # Code for CAM

# In[15]:


import scipy as sp
import cv2


# In[16]:


gap_model = model.layers[-1].get_weights()[0]

gap_model.shape


# In[17]:


gap_weights = model.layers[-1].get_weights()[0]
gap_weights.shape

cam_model  = Model(inputs=model.input,outputs=(model.layers[-13].output,model.layers[-1].output))
cam_model.summary()


# In[18]:


# utility function to preprocess an image and show the CAM
def convert_and_classify(image):

  # load the image
   img = cv2.imread(image)
    

  # preprocess the image before feeding it to the model
   img = cv2.resize(img, (180,180)) / 255.0
    
   #print(img) 

  # add a batch dimension because the model expects it
   tensor_image = np.expand_dims(img, axis=0)

  # get the features and prediction
   features,results = cam_model.predict(tensor_image)
    
   print('***************************** New 1 ***************************** ')   
  
  # generate the CAM
   show_cam(tensor_image, features, results)


# In[34]:


imgA = cv2.imread("C:/Users/tapom/Desktop/Research/4 - Thesis Writing/Paper 2/Doctor Verification/Lime Image Set/Covid/Fig11.jpg")
fig, ax1 = plt.subplots(1, 1)
ax1.imshow(imgA, interpolation = 'none')


# In[35]:


def show_cam(image_value, features, results):

#Displays the class activation map of an image

#  Args:
#    image_value (tensor) -- preprocessed input image with size 300 x 300
#    features (array) -- features of the image, shape (1, 37, 37, 128)
#    results (array) -- output of the sigmoid layer


# there is only one image in the batch so we index at `0`
    features_for_img = features[0]
    prediction = results[0]

# there is only one unit in the output so we get the weights connected to it
    class_activation_weights = gap_weights[:,0]

    print('******2')
# upsample to the image size
    class_activation_features = sp.ndimage.zoom(features_for_img, (180/9, 180/9, 1), order=2)
#spline interpolation of order = 2 (G search)
  
    print('******3')
# compute the intensity of each feature in the CAM
    cam_output  = np.dot(class_activation_features,class_activation_weights)

    print('******4')
# visualize the results
    print(f'sigmoid output: {results}')
    #print(f"prediction: {'dog' if round(results[0][0]) else 'cat'}")
    plt.figure(figsize=(8,8))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
#    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.show()

#Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. - tf.squeeze


# In[36]:


convert_and_classify("C:/Users/tapom/Desktop/Research/4 - Thesis Writing/Paper 2/Doctor Verification/Lime Image Set/Covid/Fig11.jpg")


# In[ ]:





# In[ ]:





# In[37]:


def show_cam(image_value, features, results):

#Displays the class activation map of an image

#  Args:
#    image_value (tensor) -- preprocessed input image with size 300 x 300
#    features (array) -- features of the image, shape (1, 37, 37, 128)
#    results (array) -- output of the sigmoid layer


# there is only one image in the batch so we index at `0`
    features_for_img = features[0]
    prediction = results[0]

# there is only one unit in the output so we get the weights connected to it
    class_activation_weights = gap_weights[:,0]

    print('******2')
# upsample to the image size
    class_activation_features = sp.ndimage.zoom(features_for_img, (180/9, 180/9, 1), order=2)
#spline interpolation of order = 2 (G search)
  
    print('******3')
# compute the intensity of each feature in the CAM
    cam_output  = np.dot(class_activation_features,class_activation_weights)

    print('******4')
# visualize the results
    print(f'sigmoid output: {results}')
    #print(f"prediction: {'dog' if round(results[0][0]) else 'cat'}")
    plt.figure(figsize=(8,8))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.show()

#Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. - tf.squeeze


# In[38]:


# utility function to preprocess an image and show the CAM
def convert_and_classify(image):

  # load the image
   img = cv2.imread(image)
    

  # preprocess the image before feeding it to the model
   img = cv2.resize(img, (180,180)) / 255.0
    
   #print(img) 

  # add a batch dimension because the model expects it
   tensor_image = np.expand_dims(img, axis=0)

  # get the features and prediction
   features,results = cam_model.predict(tensor_image)
    
   print('***************************** New 1 ***************************** ')   
  
  # generate the CAM
   show_cam(tensor_image, features, results)


# In[39]:


convert_and_classify("C:/Users/tapom/Desktop/Research/4 - Thesis Writing/Paper 2/Doctor Verification/Lime Image Set/Covid/Fig11.jpg")


# In[ ]:





# In[ ]:




