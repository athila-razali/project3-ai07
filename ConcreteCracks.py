# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 00:20:45 2022

@author: user

Project 3 : Concretes Cracks Classification
Goal: To perform image classification to classify concretes with or without cracks.

"""
#1. Import necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, callbacks
import pathlib
import matplotlib.pyplot as plt
import datetime
import os

#%%

#2. Data preparation
#2a. Load image files
file_path = r"C:\Users\user\Desktop\AI07\Projects\P3\Dataset"
data_dir = pathlib.Path(file_path)
SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16
train_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='training',
                                                            seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='validation',
                                                          seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)

#%%

#2b. Further split validation set, to obtain validation and test data
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#%%

#The dataset is in BatchDataset, now convert it into PrefetchDataset

AUTOTUNE = tf.data.AUTOTUNE
train_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%

#3. Define model
#3a. Data Augmentation layers
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%

#3b. Creating Feature Extraction layers
#Prepare the deep learning model by applying transfer learning
#Create layer for input preprocessing
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
#Create the base model by using MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,
                                      weights='imagenet')
#Freeze the layers in base model
base_model.trainable = False
base_model.summary()

#%%

#3c. building the classification layers here
class_names = train_dataset.class_names
nClass = len(class_names)

global_avg = layers.GlobalAveragePooling2D()
#add an output layer for classification
output_layer = layers.Dense(nClass, activation = 'softmax')

#%%

#3d. Create the entire model with functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg(x)
# x = layers.Dropout(0.3)(x) #if overfit
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#%%

#3e. Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%

#Evaluate model before model training
loss0, accuracy0 = model.evaluate(val_pf)

print("------------------------------Before Training-------------------------")
print("Loss = ", loss0)
print("Accuracy = ", accuracy0)

#%%

import datetime
#Define the Tensorboard callbacks
base_log_path = r"C:\Users\user\Desktop\AI07\DeepLearning\tb_logs"
log_path = os.path.join(base_log_path, 'project3', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%

#4. Perform model training
EPOCHS = 10
history = model.fit(train_pf, validation_data=val_pf, epochs=EPOCHS, callbacks=[tb])

#%%

#5. fine-tuning some layers in the base model
#Make the entire base_model trainable
base_model.trainable = True

#Freeze the earlier layers
for layer in base_model.layers[:100]:
    layer.trainable = False
    
base_model.summary()

#%%

#compile the model
optimizer = keras.optimizers.RMSprop(learning_rate = 0.00001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()

#%%

#Continue the model training from the previous checkpoint
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

#Follow up from previous model training
history_fine = model.fit(train_pf, validation_data=val_pf, epochs=total_epoch,
                         initial_epoch=history.epoch[-1], callbacks=[tb])

#%%

#Evaluate the model after training
test_loss, test_accuracy = model.evaluate(test_pf)

print("------------------------------After Training-------------------------")
print("Loss = ", test_loss)
print("Accuracy = ", test_accuracy)

#%%

#Deploy the model to make predictions
image_batch, label_batch = test_pf.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch), axis=1)

#%%

#Compare label vs predictions
label_vs_pred = np.transpose(np.vstack((label_batch, predictions)))
#%%
