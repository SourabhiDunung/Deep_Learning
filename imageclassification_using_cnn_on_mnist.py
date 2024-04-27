
#import libraries
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

"""#Load MNIST dataset"""

(trainX,trainY),(testX,testY)=tf.keras.datasets.mnist.load_data()

"""#split data as we have to use sklearn library"""

x= np.concatenate((trainX,testX))
y = np.concatenate((trainY,testY))
train_size = 0.8
trainX,testX,trainY,testY= train_test_split(x,y, train_size=train_size)

"""#Check the shape of input image and label(X,Y)"""

trainX.shape
testX.shape
plt.imshow(testX[9999])
testY[9999]

"""#use one hot encoding to convert 10 classes(0-9) into binary value related to class are 1 and not related is 0."""

trainY = tf.keras.utils.to_categorical(trainY, num_classes = 10)
testY = tf.keras.utils.to_categorical(testY, num_classes = 10)
trainY[34]

"""#define sequential keras model(one input and one output tensor)"""

model = tf.keras.models.Sequential()

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

"""#Layers (3)"""

model.add(Conv2D(64,kernel_size=3, padding="same", activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D())

model.add(Conv2D(32,kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(16,kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D())

"""#Flattening"""

model.add(Flatten())

"""#Add Fully connected last layer"""

model.add(Dense(10,activation="softmax"))

"""#summary of the model"""

model.summary()

"""#Complie the model"""

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics='accuracy')

"""#check for epochs-model.fit"""

model.fit(trainX,trainY,validation_data=(testX,testY),epochs=5)

"""#Test/Predict It"""

model.predict(testX[:2])

testY[:2]

plt.imshow(testX[0])
