#!/usr/bin/env python
# coding: utf-8

# # Project 4
# 
# ## Part 1: Image Classification
# 
# We will use the Fasion MNIST dataset available in torchvision.datasets.FashionMNIST for training our model. Fashion MNIST has 10 classes and 60000 train + validation images and 10000 test images. The architecture of your neural network should be 2 conv layers, 32 channels each, where each conv layer will be followed by a ReLU followed by a maxpool. This should be followed by 2 fully connected networks. Apply ReLU after the first fc layer (but not after the last fully connected layer). You should play around with different kinds of non linearities and differetn number of channels to improve your result.
# 
# Results: Once you have trained your model, show the following results for the network:
#  - Plot the train and validation accuracy during the training process.
#  - Compute a per class accuracy of your classifier on the validation and test dataset. Which classes are the hardest to get? Show 2 images from each class which the network classifies correctly, and 2 more images where it classifies incorrectly.
#  - Visualize the learned filters.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# import data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[124]:


catagories = ['shirt', 'pants', 'sweater', 'dress', 'coat', 'sandle', 'dress_shirt', 'sneaker', 'bag', 'boot']
train_images, test_images = train_images.astype('float32')/255, test_images.astype('float32')/255


# In[13]:


plt.imshow(train_images[0], cmap='gray')


# In[93]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
_, img_height, img_width, _ = train_images.shape

model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape = (img_height, img_width, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])


# In[115]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


# In[95]:


model.summary()


# In[125]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32

y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(test_labels)

train_image_generator = ImageDataGenerator() 
validation_image_generator = ImageDataGenerator() 

train_gen = train_image_generator.flow(train_images, y_train, batch_size = BATCH_SIZE)
test_gen =validation_image_generator.flow(test_images, y_test, batch_size=BATCH_SIZE)


# In[126]:


model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()


# In[127]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[128]:


#y_train = tf.keras.utils.to_categorical(train_labels)
#y_test = tf.keras.utils.to_categorical(test_labels)
epochs = train_images.shape[0]//BATCH_SIZE
history = model.fit(train_gen, validation_data=test_gen, epochs=epochs)


# In[129]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


# In[140]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(test_labels)


# In[141]:


model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()


# In[142]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[143]:


model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_test, y_test))


# In[ ]:




