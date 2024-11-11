import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
from keras.utils import to_categorical
import os
import cv2 # computer vision to treat images 
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import ResNet50,MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import glob as gb
import seaborn as sns
sns.set(style="whitegrid")

#reed , scaling ,cooding dictionary for data 

train = tf.keras.utils.image_dataset_from_directory("D:/train data",image_size = (224, 224),batch_size = 32)
test = tf.keras.utils.image_dataset_from_directory("D:/test data",image_size = (224, 224),batch_size = 32)

class_names = ["Density1Benign","Density1Malignant", "Density2Benign","Density2Malignant","Density3Benign","Density3Malignant","Density4Benign","Density4Malignant"]

cnn = tf.keras.models.Sequential()

# Step 1 - Convolution for feature map (feature extraction)

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

# Step 2 - Pooling (reduce size of feature map )
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Step 3 - Flattening(1d array (vector))

cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection(feed forward neural network )
cnn.add(tf.keras.layers.Dense(units=120, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=100, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=80, activation='relu'))

#(reduce layers )
cnn.add(tf.keras.layers.Dropout(rate=0.5))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=8, activation='softmax')) #classifier for fully conected 

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# model summary
print('Model Details are : ')
print(cnn.summary())

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train, validation_data = test, epochs =15)

# Evaluation of cnn
cnn.evaluate(test)
#predicting the model output
images = []
for filename in os.listdir('D:\predict'):
        img = cv2.imread(os.path.join('D:\predict',filename))
        if img is not None:
            images.append(img)

plt.figure(figsize=(15,15))
for images, labels in test.take(1):
  classifications = cnn(images) 
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = np.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    