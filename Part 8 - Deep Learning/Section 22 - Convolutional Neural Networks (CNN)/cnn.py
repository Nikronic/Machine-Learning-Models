# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:48:03 2018

@author: Mohammad Doosti Lakhani
"""


"""Important Note: This implementation take about hours on CPU. Use GPU or colab.research.google.com"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten


# Preprocessing
"""In this step, we just need to put training and test files to the folders with below structure :
    -test_set
        ->class1
        ->class2
        ->...
    -training_set
        ->class1
        ->class2
        ->...
    """

    
# Defining some control parameters    
image_x = 128
image_y = 128
image_channels = 3

training_set_path = 'dataset/training_set'
test_set_path = 'dataset/test_set'
    
model = Sequential() # Building sequential model

# First layer of convolution
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(image_x,image_y,image_channels)))
model.add(MaxPool2D(strides=(2,2), pool_size=(2,2)))

# Second layer of convolution
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu'))
model.add(MaxPool2D(strides=(2,2), pool_size=(2,2)))

# Flatten convolved tensors for feeding as input to Fully Connected layer
model.add(Flatten())

# First Hidden layer of fully connected
model.add(Dense(units=256, activation='relu'))

# Second Hidden layer of fully connected
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Data Augmentation
# Because we have small dataset, to prevent overfitting and train better, we use this method
# We do augmentation on both test_set and training_set

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(training_set_path, target_size = (image_x, image_y), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_set_path, target_size = (image_x, image_y), batch_size = 32, class_mode = 'binary')

# Firring model with original and augmented data
model.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)

