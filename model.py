# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:08:36 2017

@author: Yi Zhu
"""

import csv
import cv2
import numpy as np
lines=[]
with open('C:/Users/Yi Zhu/Desktop/self-driving_car/CarND-Behavioral-Cloning-P3/Self_data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]
for line in lines:
    source_path= line[0]
    filename= source_path.split('\\')[-1]
    current_path= 'C:/Users/Yi Zhu/Desktop/self-driving_car/CarND-Behavioral-Cloning-P3/Self_data/IMG/'+filename
    image=cv2.imread(current_path)
    images.append(image)
 #  images.append(cv2.flip(image,1)) #
    measurement=float(line[3])
    measurements.append(measurement)
  #  measurements.append(measurement*-1.0)#
X_train=np.array(images)
Y_train=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D()) 
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D()) #
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2,shuffle=True, nb_epoch=3)
model.save('model.h5')
