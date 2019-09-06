# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:34:00 2019

@author: Dell
"""
import tensorflow as tf
from tensorflow import keras
from model import create_model
from dataset_tf import dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


activation_fn="elu"
Batch_norm=False
kernel_init="he_normal"
dropout_rate=0.5
target_class=3
checkpoint_path="weights\my_weights.ckpt"
classes=['Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight']

image1='dataset_itr3/demo_images/image_1.jpg'
image2='dataset_itr3/demo_images/image_2.jpg'


model=create_model(input_shape=[256,256,3],activation_fn=activation_fn,Batch_norm=Batch_norm,
                   kernel_init=kernel_init,
                   dropout_rate=dropout_rate,target_class=target_class)

img1=mpimg.imread(image1)
img2=mpimg.imread(image2)
plt.figure(1)
imgplot = plt.imshow(img1)
plt.axis('off');
plt.title('Tomato healthy')

plt.figure(2)
imgplot = plt.imshow(img2)
plt.axis('off')
plt.title('Tomato Early Blight')

img=np.array([img1,img2])
model.load_weights(checkpoint_path)
pred=model.predict(img)
print(classes[0],classes[1],classes[2])
print(pred)
