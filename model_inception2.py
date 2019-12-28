# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:52:54 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 00:26:05 2019

@author: Dell
"""

import tensorflow as tf
from tensorflow import keras

def create_model(input_shape,activation_fn,Batch_norm,kernel_init,dropout_rate,target_class):
    input_layer=keras.layers.Input(shape=input_shape)
    layer_1=keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(input_layer)
    layer_2=keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(layer_1)
    pool_1=keras.layers.MaxPool2D(pool_size=2)(layer_2)
    layer_3=keras.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(pool_1)
    inception_1=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(layer_3)
    inception_1a=keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(inception_1)
    inception_1b=keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(inception_1a)
    inception_2=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(layer_3)
    inception_2a=keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(inception_2)
    inception_3=keras.layers.MaxPool2D(pool_size=2)(layer_3)
    inception_3a=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(inception_3)
    inception_4=keras.layers.Conv2D(filters=32,kernel_size=1,strides=2,padding='same'
                                ,kernel_initializer=kernel_init)(layer_3)
    concat=keras.layers.Concatenate()([inception_1b,inception_2a,inception_3a,inception_4])
    layer_4=keras.layers.Conv2D(filters=128,kernel_size=5,strides=1,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(concat)
    
    _inception_1=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(layer_4)
    _inception_1a=keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(_inception_1)
    _inception_1b=keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(_inception_1a)
    _inception_2=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(layer_4)
    _inception_2a=keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(_inception_2)
    _inception_3=keras.layers.MaxPool2D(pool_size=2)(layer_4)
    _inception_3a=keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='same'
                                ,kernel_initializer=kernel_init,activation=activation_fn)(_inception_3)
    _inception_4=keras.layers.Conv2D(filters=32,kernel_size=1,strides=2,padding='same'
                                ,kernel_initializer=kernel_init)(layer_4)
    concat=keras.layers.Concatenate()([_inception_1b,_inception_2a,_inception_3a,_inception_4])
    
    
    layer_5=keras.layers.Conv2D(filters=128,kernel_size=5,strides=2,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(concat)
    layer_6=keras.layers.Conv2D(filters=256,kernel_size=7,strides=2,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(layer_5)
    layer_7=keras.layers.Conv2D(filters=256,kernel_size=7,strides=2,padding='same',
                                kernel_initializer=kernel_init,activation=activation_fn)(layer_6)
    flat=keras.layers.Flatten()(layer_7)
    dense1=keras.layers.Dense(512,activation='relu',kernel_initializer=kernel_init)(flat)
    drop_1=keras.layers.Dropout(rate=dropout_rate)(dense1)
    dense2=keras.layers.Dense(256,activation='relu',kernel_initializer=kernel_init)(drop_1)
    drop_2=keras.layers.Dropout(rate=dropout_rate)(dense2)
    dense3=keras.layers.Dense(128,activation='relu',kernel_initializer=kernel_init)(drop_2)
    drop_3=keras.layers.Dropout(rate=dropout_rate)(dense3)
    output=keras.layers.Dense(target_class,activation='softmax')(drop_3)
    
    model=keras.Model(inputs=[input_layer],outputs=[output])
    return model
    
    