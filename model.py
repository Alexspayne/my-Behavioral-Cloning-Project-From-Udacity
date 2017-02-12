import numpy as np
from scipy import ndimage
import csv
from os import path
import os
# Fix error with TF and Keras (Taken from Keras Lab)
import tensorflow as tf
tf.python.control_flow_ops = tf

datasets = os.listdir('./data')
#Getting data from the first run.
images = []
measurements = []
for folder in datasets:
    with open('data/' + folder + '/driving_log.csv') as csvfile:
        file_paths = []
        steering_angles = []
        logreader = csv.DictReader(csvfile,fieldnames=['center','left','right','steering','throttle','brake','speed'])
        for row in logreader:
            file_paths.append(row['center'])
            steering_angles.append(row['steering'])

    for image_index in range(len(file_paths)):
        image_array = ndimage.imread("".join(['data/' + folder + '/IMG/',path.basename(file_paths[image_index])]))
        images.append(image_array)
        angle = steering_angles[image_index]
        measurements.append(angle)


# Convert to an ndarray
X_train = np.array(images)
y_train = np.array(measurements)

# For testing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
# Taking a shot in the dark with this one: (Haven't read the doc)
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# TODO: Build a Multi-layer feedforward neural network with Keras here.

# Normalization layer will go here once I figure that out.

# Thanks to David for instruction on normalization using lambda layers
model.add(Cropping2D(cropping=((50,25), (10,10)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x /255.0 - 0.5))
# Conv layers exactly from paper

model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(36,5,5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(48,5,5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64,3,3, activation='relu'))

model.add(Convolution2D(64,3,3, activation='relu'))

# Flatten after all the conv layers
model.add(Flatten())

# Fully connected layers exactly as the paper had.
# They didn't show any activations...?
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))


#The final layer has an output of 1, because we want one steering angle
model.add(Dense(1))

# I don't think this is necessary but I'll leave it here incase.
# model.add(Activation('softmax'))

model.compile('adam', 'mse', ['accuracy'])

#Defining my generator right here because why not.
import random

def batch_generator(inputs, targets):
    # samples = zip(inputs, targets)
    # I guess zip by itself doesn't work
    samples = [(x,y) for x, y in zip(inputs, targets)]
 #   sample_size = 500
    while True:
  #      batch = []
   #     for _ in range(sample_size):
        yield random.choice(samples)
        
        
history = model.fit(X_train, y_train, batch_size=216, nb_epoch=4, validation_split=0.2)
# generator = batch_generator(X_train, y_train)  
# history = model.fit_generator(generator,
#                               samples_per_epoch=3000,
#                               nb_epoch=4)
                              
model.save('model.h5')

