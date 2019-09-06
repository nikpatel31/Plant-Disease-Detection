# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:54:19 2019

@author: Dell
"""
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class dataset:
    def __init__(self,folder_name,repeat=1,shuffle_buffer_size=1000,batch_size=10,n_parse_threads=2):
        self.folder_name=folder_name
        self.repeat=repeat
        self.shuffle_buffer_size=shuffle_buffer_size
        self.batch_size=batch_size
        self.n_parse_threads=n_parse_threads
        self.DATASET_PATH="dataset_itr3/"
        #self.classes=os.listdir(os.path.join(self.DATASET_PATH))
        self.classes=os.listdir(self.DATASET_PATH+self.folder_name)
        
    def preprocess(self,image):
        image=tf.image.decode_jpeg(image,channels=3)
        image=tf.image.random_brightness(image,0.3)
        image=tf.image.random_flip_left_right(image)
        image=tf.image.random_flip_up_down(image)
        image/=255
        #image=2*image-1
        return image
    
    def load_preprocess(self,path):
        image=tf.io.read_file(path)
        return self.preprocess(image)
    
    def image_list(self):
        self.train_path=os.path.join(self.DATASET_PATH,self.folder_name)
        for i in range(len(self.classes)):
            class_path=os.path.join(self.train_path,self.classes[i])
            img_list=os.listdir(class_path)
            for j in range(len(img_list)):
                img_list[j]=os.path.join(class_path,img_list[j])
                
            if(i==0):
                self.image_path_list=np.array(img_list)
            else:
                self.image_path_list=np.append(self.image_path_list,img_list)
                
    def label_list(self):
        for i in range(len(self.classes)):
            class_path=os.path.join(self.train_path,self.classes[i])
            label_len=len(os.listdir(class_path))
            if(i==0):
                self.label_list=np.zeros((label_len,1),dtype=int)
            else:
                self.label_list=np.append(self.label_list,(np.zeros((label_len,1),dtype=int)+i))
        encoder=LabelBinarizer()
        self.label_list_new=encoder.fit_transform(self.label_list)
        
    def preprocess_load_ds(self,image_path_list,label_list_new):
        return self.load_preprocess(image_path_list),label_list_new
    
    def dataset_ready(self):
        self.image_list()
        self.label_list()
        
        image_path_ds=tf.data.Dataset.from_tensor_slices(self.image_path_list)
        label_path_ds=tf.data.Dataset.from_tensor_slices(tf.cast((self.label_list_new),tf.int64))
    
        ds=tf.data.Dataset.zip((image_path_ds,label_path_ds))
        image_label_ds=ds.map(self.preprocess_load_ds,num_parallel_calls=self.n_parse_threads)
        
        
    
        ds=image_label_ds.shuffle(buffer_size=self.shuffle_buffer_size)
        ds=ds.repeat()
        ds=ds.batch(self.batch_size)
        ds=ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
        return(ds)