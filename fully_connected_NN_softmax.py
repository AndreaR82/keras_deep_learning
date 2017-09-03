"""
Created on Sat Jul 29 09:54:37 2017

@author: andrea

Description
Example of fully connected neural network using Keras and GPU.
Data is randomly generated
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import keras
import os



# Generate dummy data
n_train = 100000
n_test = 20000
n_input = 200
x_train = np.random.random((n_train, n_input))
y_train = keras.utils.to_categorical(np.random.randint(10,                                                             size=(n_train, 1)), num_classes=10)
x_test = np.random.random((n_test, n_input))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(n_test, 1)), num_classes=10)


# Build net
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(1000, activation='relu', input_dim=n_input))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=1048)
score = model.evaluate(x_test, y_test, batch_size=128)

