import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
import cv2
import csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, merge, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D

print('Done with importing')

# Read in the recorded data from driving_log
lines = []
with open('/home/workspace/img/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Extracting X_train and y_train        
images = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/home/workspace/img/test/'+filename
    image = ndimage.imread(current_path)
    images.append(image)
headers = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed' ]
data= pd.read_csv('/home/workspace/img/driving_log.csv',names = headers)
data  = np.array(data)
X_train = np.array(images)
y_train = data[:,3]


# # data preprocessing

# find the left and right turn angles:
# For angles<0(left turns), the images and corresponding 
# angles are denoted as X_low, y_low;
# For angles>0(right turns), the images and corresponding 
# angles are denoted as X_high, y_high;
X_high = []
y_high = []
X_low  = []
y_low = []
# Threshold for spliting X_low and X_high
ang_low = -0.09090909090909083
ang_high = 0.09090909090909083

# right turning angles
for idx in range(len(y_train)):
    if ((y_train[idx]>ang_high)):
        X_high.append(X_train[idx])
        y_high.append(y_train[idx])

# left turning angles        
for idx in range(len(y_train)):
    if ((y_train[idx]<ang_low)):
        X_low.append(X_train[idx])
        y_low.append(y_train[idx])

X_high = np.array(X_high)
y_high = np.array(y_high)
X_low = np.array(X_low)
y_low = np.array(y_low)


# # data augmentation

# data generation by copying and concatenate the given image and angle set to the original one for n times
def generate_img(img,ang,num):
    for i in range(num):
        img = np.concatenate((img,img),axis=0)
        ang = np.concatenate((ang,ang),axis=0)
    return img, ang



it_low = 3 # num of image iterations
it_high = 4 # num of image iterations
X_lownew, y_lownew = generate_img(X_low,y_low,it_low)#generated image and angle set for left turns
X_highnew, y_highnew = generate_img(X_high,y_high,it_high)#generated image and angle set for right turns

# Concatenate original image set and angle set with generated ones
X_tabnew = np.concatenate((X_lownew,X_highnew),axis=0)
y_tabnew = np.concatenate((y_lownew,y_highnew),axis=0)
X_aug = np.concatenate((X_train,X_tabnew),axis=0)
y_aug = np.concatenate((y_train,y_tabnew),axis=0)


# split training, validation and testing sets
X_aug,y_aug = shuffle(X_aug,y_aug)
X_train, X_valid, y_train, y_valid = train_test_split(X_aug,y_aug,random_state=0, test_size=0.2)

# Generator
import sklearn

def generator(X_data,y_data, batch_size=32):
    num_samples = len(y_data)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_train = X_data[offset:offset+batch_size]
            y_train = y_data[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_train, y_train)
            
batch_size =32
train_set = generator(X_train,y_train,batch_size = batch_size)
valid_set = generator(X_valid,y_valid,batch_size = batch_size)


# Model Archetecture of the neuro networks 
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))                              
model.add(Conv2D(32,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(64,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(128,(5,5),activation = "relu", strides=(2,2)))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Conv2D(32,(3,3))) 
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1))  
print(model.summary())
model.compile(loss = 'mse',optimizer='adam')
model.fit_generator(train_set,
            steps_per_epoch=np.ceil(len(y_train)/batch_size), 
            validation_data=valid_set, 
            validation_steps=np.ceil(len(y_valid)/batch_size),
            epochs=5, verbose=1)
model.save('model.h5')
print('model saved')