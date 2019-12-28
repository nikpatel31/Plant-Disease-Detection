# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:20:15 2019

@author: Dell
"""

'''Import dependencies'''

from tensorflow import keras
import tensorflow as tf 
import numpy as np
import os
#from nik_model_colabs import create_model
from model_inception2 import create_model
from dataset_tf import dataset
from math import exp
#from clr_callback import CyclicLR

'''Hypertuning'''
learning_rate=1e-5
batch_size=8
momentum=0.9
nesterov=True
metrics=[keras.metrics.categorical_accuracy]
epochs=30
valid_step=272
train_step=2228
epoch_step=50    #no of epoch after which lr decays
lr_step=exp(-1)    #factor by which lr decays
activation_fn="elu"
Batch_norm=False
kernel_init="he_normal"
dropout_rate=0.5
target_class=20
loss_list=np.array([])




''' Class weights '''
class_weights={0: 34.88,1: 35.39,2: 79.90,3: 13.36,4: 42.84,5: 18.43,6: 18.91,7: 22.31,8: 21.978,9: 144.592,
                        10: 21.978,11: 10.332,12: 21.978,13: 13.81,14: 11.51 , 15: 23.08,16: 12.4,17: 13.11,
                        18: 15.65,19: 58.92}

'''Function to create model in Keras'''
'''
model=create_model(input_shape=[256,256,3],activation_fn='relu',Batch_norm=False,
                   kernel_init='he_normal',dropout_rate=dropout_rate,target_class=target_class)

'''
model=tf.keras.models.load_model("checkpoint/my_model_v1.h5")
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

'''
class my_callback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None,learning_rate=learning_rate):
        learning_rate= learning_rate*exp(batch)
        loss_list=np.append(loss_list,[])
'''        
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath="checkpoint/my_model_v2.h5",verbose=1,save_weigths_only=True)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
#clr=CyclicLR(base_lr=1e-5,max_lr=1e-2,step_size=3200,mode="triangular2")


'''model complier and optimizer'''
optimizer=keras.optimizers.Nadam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=metrics)


'''Dataset Create'''
obj=dataset("train",batch_size=batch_size)
train_ds=obj.dataset_ready()
obj=dataset("valid",batch_size=batch_size)
valid_ds=obj.test_dataset_ready()


'''Training'''

history=model.fit(train_ds,epochs=epochs,validation_data=valid_ds,validation_steps=valid_step,callbacks=[tensorboard_cb,lr_scheduler,checkpoint_cb],
                 steps_per_epoch=train_step)
'''
history=model.fit(train_ds,epochs=epochs,callbacks=[tensorboard_cb,clr,checkpoint_cb],
                 steps_per_epoch=train_step)
'''

    
model.save_weights("weights_v2\my_weights.ckpt")
#model.evaluate(valid_ds,steps=valid_step)
    