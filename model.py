# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:24:02 2019

@author: Dell
"""
import tensorflow as tf
from tensorflow import keras

def create_model(input_shape,activation_fn,Batch_norm,kernel_init,dropout_rate,target_class):
 
    model=keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(filters=32,kernel_size=7,strides=1,padding="same",input_shape=input_shape,
                        kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    model.add(keras.layers.Conv2D(filters=32,kernel_size=7,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Conv2D(filters=64,kernel_size=5,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    model.add(keras.layers.Conv2D(filters=64,kernel_size=5,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
        
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding="same",kernel_initializer=kernel_init))
    if(Batch_norm):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation_fn))
    else:
        model.add(keras.layers.Activation(activation_fn))
        
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512,activation=activation_fn))
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(256,activation=activation_fn))
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(64,activation=activation_fn))
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(target_class,activation="softmax"))
    
    return model
    
    
