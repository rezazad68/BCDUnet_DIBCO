# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza winchester
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import models as M
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks
import pickle

batch_size = 8

####################################  Load Data #####################################3
patches_imgs_train  = np.load('patches_image.npy')
patches_masks_train = np.load('patches_masks.npy')

patches_imgs_train  /= 255.
patches_masks_train /= 255.

patches_imgs_train  = np.expand_dims(patches_imgs_train,  axis = 3)
patches_masks_train = np.expand_dims(patches_masks_train, axis = 3)

print('Dataset Prepared')

model = M.BCDU_net_D3(input_size = (128, 128, 1))
model.summary()

print('Training')

nb_epoch = 100

mcp_save = ModelCheckpoint('weight_text.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(patches_imgs_train,patches_masks_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, reduce_lr_loss] )

print('Trained model saved')
with open('hist_dibco', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


