# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:20:15 2019

@author: Dell
"""

'''Import dependencies'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from model import create_model
from dataset_tf import dataset

'''Hypertuning'''
learning_rate=0.0001
batch_size=16
momentum=0.9
nesterov=True
metrics=[keras.metrics.categorical_accuracy]
epochs=20
valid_step=int(449/batch_size)
train_step=512
epoch_step=5    #no of epoch after which lr decays
lr_step=0.01    #factor by which lr decays
activation_fn="elu"
Batch_norm=False
kernel_init="he_normal"
dropout_rate=0.5
target_class=3



'''Function to create model in Keras'''
model=create_model(input_shape=[256,256,3],activation_fn=activation_fn,Batch_norm=Batch_norm,
                   kernel_init=kernel_init,
                   dropout_rate=dropout_rate,target_class=target_class)


'''Directory for tf_logs'''
root_logdir = os.path.join(os.curdir, "tf_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()


'''callbacks'''
def exponential_decay_fn(epoch):
    return learning_rate * lr_step**(epoch / epoch_step)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath="my_model.h5",verbose=1,save_weigths_only=True)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)


'''model complier and optimizer'''
optimizer=keras.optimizers.SGD(lr=learning_rate,momentum=momentum,nesterov=nesterov)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=metrics)


'''Dataset Create'''
obj=dataset("train",batch_size=batch_size)
train_ds=obj.dataset_ready()
obj=dataset("valid",batch_size=batch_size)
valid_ds=obj.dataset_ready()


'''Training'''
history=model.fit(train_ds,epochs=epochs,validation_data=valid_ds,validation_steps=valid_step,callbacks=[tensorboard_cb,lr_scheduler,checkpoint_cb],
                 steps_per_epoch=train_step)

model.save_weights("weights\my_weights.ckpt")