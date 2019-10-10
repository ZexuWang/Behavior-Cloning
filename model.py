
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

lines = []
with open('/home/workspace/img/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# Extracting X_train and y_train        
images = []
for lin in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/home/workspace/img/test/'+filename
    image = ndimage.imread(current_path)
    images.append(image)
headers = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed' ]
data = pd.read_csv('/home/workspace/img/driving_log.csv',names = headers)
data = np.array(data)
X_train = np.array(images)
y_train = data[:,3]


# data preprocessing

# find the left and right turn angles
X_high = []
y_high = []
X_low  = []
y_low = []
ang_low = -0.09090909090909083
ang_high = 0.09090909090909083
# right angles
for idx in range(len(y_train)):
    if ((y_train[idx]>ang_high)):
        X_high.append(X_train[idx])
        y_high.append(y_train[idx])

# left angles        
for idx in range(len(y_train)):
    if ((y_train[idx]<ang_low)):
        X_low.append(X_train[idx])
        y_low.append(y_train[idx])

X_high = np.array(X_high)
y_high = np.array(y_high)
X_low = np.array(X_low)
y_low = np.array(y_low)


# data augmentation
def generate_img(img,ang,num):
    for i in range(num):
        img = np.concatenate((img,img),axis=0)
        ang = np.concatenate((ang,ang),axis=0)
    return img, ang



it_low = 4 # num of iterations
it_high = 5 # num of iterations
X_lownew, y_lownew = generate_img(X_low,y_low,it_low)
X_highnew, y_highnew = generate_img(X_high,y_high,it_high)

X_tabnew = np.concatenate((X_lownew,X_highnew),axis=0)
y_tabnew = np.concatenate((y_lownew,y_highnew),axis=0)
X_aug = np.concatenate((X_train,X_tabnew),axis=0)
y_aug = np.concatenate((y_train,y_tabnew),axis=0)


# split training, validation and testing sets
X_aug,y_aug = shuffle(X_aug,y_aug)
X_train, X_valid, y_train, y_valid = train_test_split(X_aug,y_aug,random_state=0, test_size=0.2)

# Model Archetecture of the neuro networks using transfer learning
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3


img_shape = (160,320,3)

# Tensor input
tensor_input = Input(shape=(160,320,3))
# resize the input using a lambda layer
##resized_input = Lambda((lambda x: x/255.0 -0.5))(tensor_input)

inception = InceptionV3(weights='imagenet', include_top=False,
                        input_shape=img_shape)

# Feeds the re-sized input into Inception Model
x = inception.output
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1)(x)


model = Model(input=inception.input, output=x)
model.compile(loss = 'mse',optimizer='adam')
model.fit(X_train,y_train,epochs=3,validation_data=(X_valid,y_valid))
model.save('model.h5')
print('model saved')