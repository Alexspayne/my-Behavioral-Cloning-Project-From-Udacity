import numpy as np
from scipy import ndimage
import csv
import os
from os import path

# Fix error with TF and Keras (Taken from Keras Lab)
import tensorflow as tf
tf.python.control_flow_ops = tf

# I saved my training data into different folders to better manage it.
# This made it easier to discard a bad data when I was collecting.
datasets = os.listdir('./data')

images = []
measurements = []

# Here I loop though each folder and load the data into memory.
for folder in datasets:
    with open('data/' + folder + '/driving_log.csv') as csvfile:
        file_paths = []
        steering_angles = []
        logreader = csv.DictReader(csvfile,fieldnames=['center',
                                                       'left',
                                                       'right',
                                                       'steering',
                                                       'throttle',
                                                       'brake',
                                                       'speed'])
        
        # Here I'm extracting just the columns I'm interested in for training.
        for row in logreader:
            file_paths.append(row['center'])
            steering_angles.append(row['steering'])

    for image_index in range(len(file_paths)):
        image_array = ndimage.imread(
            "".join(['data/' + \
                     folder + \
                     '/IMG/',
                     path.basename(file_paths[image_index])]))
        # Add the sample to the array which will be the training data.
        images.append(image_array)
        angle = steering_angles[image_index]
        measurements.append(angle)
        #Also add the left/right flipped image.
        images.append(np.fliplr(image_array))
        measurements.append(-1*float(angle))


# Convert to an ndarray
X_train = np.array(images)
y_train = np.array(measurements)

# Free up some memory by offering up the intermediate training data variables
# for garbage collection.  I'm not sure if this is necessary; it doesn't hurt.
images = measurements = image_array = None

# Building the model architecture
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# This cropping layer was give as a tip in the lesson.
# I further cropped the image because I figured that it would help keep
# the model focused just on the road.
model.add(Cropping2D(cropping=((50,25), (10,10)), input_shape=(160,320,3)))

# Thanks to David for instruction on normalization using lambda layers
model.add(Lambda(lambda x: x /255.0 - 0.5))

# Five convolutional layers similar to NVidia's model.
# I only added pooling on the first three layers due to memory contraints.

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

# Fully connected layers exactly as the NVidia paper had.
# I added a couple dropout layers to help generalize the model.
model.add(Dense(100, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(10, activation='relu'))

#The final layer has an output of 1, because we want one steering angle
model.add(Dense(1))

# I'm using the adam optimizer. 
model.compile('adam', 'mse', ['accuracy'])

# David used 4 epochs in his Q&A.  I've found that to be enough too.
# The Keras model object has data shuffling built in, so that was easy.

history = model.fit(X_train,
                    y_train,
                    batch_size=300,
                    nb_epoch=4,
                    validation_split=0.2,
                    shuffle=True)

model.save('model.h5')
