# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:02:30 2019

@author: Not Your Computer
"""

from __future__ import print_function
import numpy as np
import random
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from supportingUDFs import getImgStack
from skimage.transform import resize
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from keras.models import Model
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

#%%
# DEFINE VARIABLES
read_images = True; normalize = True; color_imgs = True; rand_st = 1;
n_imgs = 'all'; validation_ratio = 0.05; test_ratio = 0.10; img_size = 64;

print('\nSCRIPT PARAMETERS')
print('------------------------------------------')
print('normalize:          {}'.format(normalize))

if color_imgs:
    print('Colors:             RGB'.format(color_imgs))
else:
    print('Colors:             GrS'.format(color_imgs))

print('img_size:           {} x {}'.format(img_size, img_size))

host_dir = 'C:\\Users\\Not Your Computer\\Desktop\\Cell_Images'
root_folder = 'cell_images'
pos_folder = 'Parasitized'
neg_folder = 'Uninfected'
pos_folder_path = '{}\\{}\\{}'.format(host_dir, root_folder, pos_folder)
neg_folder_path = '{}\\{}\\{}'.format(host_dir, root_folder, neg_folder)
save_path = '{}\\{}'.format(host_dir, 'saved_cnn.h5')

# CREATE IMAGE STACK
if read_images:
    st = time.time()
    img_stack, classes = getImgStack(pos_folder_path, neg_folder_path, img_size, color_imgs, subset = n_imgs)
    print('\nImages loaded.')
print('\n{} sec'.format(time.time()-st))

# PREPROCESS IMAGES
if normalize:
    print('\nNormalizing Images...')
    st = time.time()
    img_stack = np.divide(img_stack, np.max(img_stack))
    print('\n{} sec'.format(time.time()-st))
else:
    img_stack = np.array(img_stack)
    
if color_imgs == False:
   img_stack = img_stack.reshape(len(img_stack), img_size, img_size, 1)
   
# CREATE TRAIN/TEST/VAL SPLITS
print('\n\nSPLITTING DATA')  
st = time.time()
# Create Validation Set
print('\nSeparating Validation Data...')
x_train, x_valid, y_train, y_valid = train_test_split(img_stack, classes, test_size = validation_ratio,
                                                    random_state = rand_st, shuffle = True)

# Create Train/Test Sets
print('\nSeparating Train/Test Data')
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = test_ratio,
                                                    random_state = rand_st, shuffle = True)

print('n_imgs:             {}'.format(len(img_stack)))
print('validation_ratio:   {} ({} imgs)'.format(validation_ratio, len(x_valid)))
print('test_ratio:         {} ({} imgs)'.format(test_ratio, len(x_test)))

print('\n{} sec'.format(time.time()-st))

#%%
# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================
from supportingUDFs import CNN
activations = ['tanh', 'softmax', 'relu', 'elu', 'sigmoid', 'linear', 'exponential']
optimizers = ['sgd', 'rms', 'adagrad', 'adadelta', 'adam']
losses = ['mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'cosine_proximity']
batch_size = 128
model_size = 'small'

# BUILD AND COMPILE MODELS
print('\nBUILDING MODELS...')
cls_cnn = CNN(model_type = 'classifier', cmode = 'rgb', final_act = 'softmax',
              loss_type = 'binary_crossentropy', opt_type = 'adagrad', size = model_size,
              learn_rate = 0.0045, img_size = img_size, eval_metric = 'categorical_accuracy')
reg_cnn = CNN(model_type = 'regressor', cmode = 'rgb',final_act = 'sigmoid',
              loss_type = 'mean_squared_error', opt_type = 'adagrad', size = model_size,
              learn_rate = 0.0045, img_size = img_size, eval_metric = 'accuracy')

# TRAIN MODELS
print('\nTRAINING CLASSIFICATION MODELS...')
print('------------------------------------------')
#cls_cnn.train(x_train, y_train, x_test, y_test, batch_size = batch_size, epochs = 5, folds = 10)
print('\nTRAINING REGRESSION MODELS...')
print('------------------------------------------')
reg_cnn.train(x_train, y_train, x_test, y_test, batch_size = batch_size, epochs = 7, folds = 10)

#%%
# VALIDATE AND EVALUATE MODELS
print('\nEVALUATING CLASSIFICATION MODELS...')
print('------------------------------------------')
#cls_cnn.evaluate(x_valid, y_valid, plots_on = True)
print('\nEVALUATING REGRESSION MODELS...')
print('------------------------------------------')
reg_cnn.evaluate(x_valid, y_valid, plots_on = True)

print('\nImage Breakdown:')
print('------------------------------------------')
print('Training Imgs:       {}'.format(len(x_train)))
print('Test Imgs:           {}'.format(len(x_test)))
print('Validation Imgs:     {}'.format(len(x_valid)))
print('Unused Imgs:         {}'.format(len(img_stack)-(len(x_train)+len(x_test)+len(x_valid))))

#%%
# SAVE MODELS
print('\nSAVING CLASSIFIER...')
#cls_cnn.save(str(cls_cnn.code)+'.h5')
print('\nSAVING REGRESSOR...')
reg_cnn.save(str(reg_cnn.code)+'.h5')
print('\n{} sec'.format(time.time()-st))
print('\nCls Model Code:  ', cls_cnn.code)
print('Reg Model Code:  ', reg_cnn.code)

#%%
# PLOT LAYERS
rand_idx = random.randint(0,250)
sample_img = img_stack[rand_idx]
sample_class = classes[rand_idx]
print(f'Image No {rand_idx}, Class {sample_class}')
#cls_cnn.plotFilters(img = sample_img, img_class = sample_class)
reg_cnn.plotFilters(img = sample_img, img_class = sample_class)