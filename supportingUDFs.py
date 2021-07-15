# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:23:41 2019

@author: Not Your Computer
"""

import os
import time
import numpy as np
from random import shuffle
from skimage.transform import resize
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from keras.models import Model
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import load_model

#%%

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def getImgStack(pos_folder_path, neg_folder_path, min_img_dim, color_imgs = True, subset = 'all'):
    pos_folder = os.path.basename(pos_folder_path)
    neg_folder = os.path.basename(neg_folder_path)
    
    img_stack = []; classes = []; path_list = []
    
    # Collect Positive Image Paths
    folder_path = pos_folder_path
    print('\nReading Image Paths from {}'.format(folder_path))
    for f in os.listdir(folder_path):
        img_path = '{}\\{}'.format(folder_path, f)
        fparts = f.split('.')
        fext = fparts[-1].lower()
    
        if os.path.isfile(img_path) and fext == 'png':
            path_list.append(img_path)
    
    # Collect Negative Image Paths
    folder_path = neg_folder_path
    print('\nReading Image Paths from {}'.format(folder_path))
    for f in os.listdir(folder_path):
        img_path = '{}\\{}'.format(folder_path, f)
        fparts = f.split('.')
        fext = fparts[-1].lower()
    
        if os.path.isfile(img_path) and fext == 'png':
            path_list.append(img_path)
        
    # Shuffle list of paths
    shuffle(path_list)
    
    if subset == 'all':
        subset = len(path_list)
    
    # Create stack of images
    i = 0
    print('\nLoading Images...')
    for path in path_list[0:subset]:
        if color_imgs:
            img = plt.imread(path)
            h,w,d = np.shape(img)
        else:
            img = plt.imread(path)
            img = rgb2gray(img)
            h,w = np.shape(img)
        
        if min(h,w) >= min_img_dim:
            
            # Resize Images to Square AR
            img_resized = resize(img, (min_img_dim, min_img_dim))
            img_stack.append(img_resized)
            
            # Determine Class
            if pos_folder in path:
                classes.append(1)
            else:
                classes.append(0)
        i += 1

    return img_stack, classes

def gridPlot6(img_stack):
    """
    A grid of 2x2 images with a single colorbar
    """
    F = plt.figure(figsize = (20,20))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (2,3), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    
    i = 0
    for img in img_stack[0:6]:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig('gplot16.png')
    if 'gplot16.png' in os.listdir():
        plt.savefig('gplot16_2.png')
#    for cax in grid.cbar_axes:
#        cax.toggle_label(False)
    return

def gridPlot12(img_stack):
    """
    A grid of 2x2 images with a single colorbar
    """
    F = plt.figure(figsize = (30,30))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (3,4), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    
    i = 0
    for img in img_stack[0:12]:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig('gplot12.png')
#    for cax in grid.cbar_axes:
#        cax.toggle_label(False)
    return

def gridPlot16(img_stack):
    """
    A grid of 2x2 images with a single colorbar
    """
    F = plt.figure(figsize = (30,30))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (4,4), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    
    i = 0
    for img in img_stack[0:16]:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig('gplot16.png')
    if 'gplot16.png' in os.listdir():
        plt.savefig('gplot16_2.png')
#    for cax in grid.cbar_axes:
#        cax.toggle_label(False)
    return


def gridPlot48(img_stack):
    """
    A grid of 2x2 images with a single colorbar
    """
    F = plt.figure(figsize = (50,50))
    F.subplots_adjust(left = 0.05, right = 0.95)
    grid = ImageGrid(F, 142, nrows_ncols = (6,8), axes_pad = 0.0, share_all = True,
                     label_mode = "L", cbar_location = "top", cbar_mode = "single")
    
    i = 0
    for img in img_stack[0:48]:
        im = grid[i].imshow(img, interpolation = "nearest", vmin = 0, vmax = 255)
        i += 1 
    grid.cbar_axes[0].colorbar(im)
    plt.savefig('gplot48.png')
#    for cax in grid.cbar_axes:
#        cax.toggle_label(False)
    return


# Plot Filter Layer Results
def plotLayer(model, layer_idx, img, cmap = 'viridis', normalize = False):
    h,w,d = np.shape(img)
    img = img * 255
    m = Model(inputs = model.input, outputs = model.get_layer(index = layer_idx).output)
    p = m.predict(np.reshape(img, (1,h,w,d)))
    img_stack = []
    p_size = np.shape(p)[3]
    for i in range(p_size):
        p_out = p[0,:,:,i]
        img_stack.append(p_out)
    if p_size == 16:
        gridPlot16(img_stack)
    elif p_size == 48:
        gridPlot48(img_stack)
    elif p_size == 12:
        gridPlot12(img_stack)
    elif p_size == 6:
        gridPlot6(img_stack)
    return


def plotROC(y_valid, y_predicted):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_valid, y_predicted)
    auc_score = auc(fpr, tpr)
    # Zoom in view of the upper left corner.
    plt.figure(figsize = (9,6))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left = 0, right = 1)
    plt.ylim(top = 1, bottom = 0)
    return auc_score


def codifyModel(m_type = '', color_type = '', opt_type = '', final_act = '', lr = '', acc = '', auc = ''):
    model_code = ''; parameter_list = []
    model_param = m_type
    if model_param != '':
        if model_param == 'classification':
            model_param = 'CLS'
        elif model_param == 'regression':
            model_param = 'REG'
        parameter_list.append(model_param)
    model_param = color_type
    if model_param != '':
        if model_param == True:
            model_param = 'RGB'
        elif model_param == False:
            model_param = 'GrS'
        parameter_list.append(model_param)
    model_param = opt_type
    if model_param != '':
        parameter_list.append(model_param)
    model_param = final_act
    if model_param != '':
        parameter_list.append(model_param)
    model_param = lr
    if model_param != '':
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[-1]
        parameter_list.append(model_param)
    model_param = acc
    if model_param != '':
        model_param = model_param * 100
        model_param = round(model_param)
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[0]
        parameter_list.append(model_param)
    model_param = auc
    if model_param != '':
        model_param = model_param * 100
        model_param = round(model_param)
        model_param = str(model_param)
        model_p_parts = model_param.split('.')
        model_param = model_p_parts[0]
        parameter_list.append(model_param)
    model_code = '_'.join(parameter_list)
    parameter_list = [m_type, color_type, opt_type, final_act, lr, acc, auc]
    return model_code

def RGBClassCNN(img_size, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = (img_size, img_size, 3), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(48, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(2, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    return model

def GrSClassCNN(img_size, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = (img_size, img_size, 1), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(48, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(2, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    return model

def RGBRegCNN(img_size, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = (img_size, img_size, 3), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(48, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu', input_dim = 64*64))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model

def GrSRegCNN(img_size, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = (img_size, img_size, 1), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(48, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu', input_dim = 64*64))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model

def SmallCNN(img_size, model_type = 'classifier', channels = 1, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(6, (3,3), input_shape = (img_size, img_size, channels), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout:
        model.add(Dropout(0.35))                            # DROP
    model.add(Conv2D(12, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 16 ==> 8
    if dropout != None:
        model.add(Dropout(0.35))                            # DROP
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(2352, activation = 'relu'))              # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))               # DENSE
    if dropout != None:
        model.add(Dropout(0.35))                            # DROP
    if model_type == 'classifier':
        model.add(Dense(2, activation = final_act))             # DENSE
    else:
        model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    if model_type == 'classifier':
        model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    else:
        model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model

def MedCNN(img_size, channels = 1, model_type = 'classifier', opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape = (img_size, img_size, channels), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(48, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(16, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(576, activation = 'relu', input_dim = 64*64))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    if model_type == 'classifier':
        model.add(Dense(2, activation = final_act))             # DENSE
    else:
        model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    if model_type == 'classifier':
        model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    else:
        model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model

def LargeCNN(img_size, model_type = 'classifier', channels = 1, opt_type = 'rms', losses = 'mean_squared_error', final_act = 'sigmoid', learn_rate = 0.001, dropout = True):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = (img_size, img_size, channels), activation = 'relu', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size = (2,2)))              # 128 ==> 64
    if dropout:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(128, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(128, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 64 ==> 32
    if dropout != None:
        model.add(Dropout(0.33))                            # DROP
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(Conv2D(64, (3,3), activation = 'relu'))       # CONV 2D
    model.add(MaxPooling2D(pool_size = (2,2)))              # 32 ==> 16
    if dropout != None:
        model.add(Dropout(0.33))  
    model.add(Flatten())                                    # FLATTEN
    model.add(Dense(4096, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    model.add(Dense(64, activation = 'relu'))             # DENSE
    if dropout != None:
        model.add(Dropout(0.40))                            # DROP
    if model_type == 'classifier':
        model.add(Dense(2, activation = final_act))             # DENSE
    else:
        model.add(Dense(1, activation = final_act))             # DENSE
    # Define Optimizer
    if opt_type == 'rms':
        opt = keras.optimizers.RMSprop(lr = learn_rate, rho = 0.9, epsilon = None, decay = 1e-6)
    if opt_type == 'sgd':
        opt = keras.optimizers.SGD(lr = learn_rate, decay = 1e-6)
    if opt_type == 'adam':
        opt = keras.optimizers.Adam(lr = learn_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    if opt_type == 'adagrad':
        opt = keras.optimizers.Adagrad(lr = learn_rate, epsilon = None, decay = 0.0)
    if opt_type == 'adadelta':
        opt = keras.optimizers.Adadelta(lr = learn_rate, rho = 0.95, epsilon = None, decay = 0.0)
    # Compile Model
    if model_type == 'classifier':
        model.compile(loss = losses, optimizer = opt, metrics = ['categorical_accuracy'])
    else:
        model.compile(loss = losses, optimizer = opt, metrics = ['accuracy'])
    return model

class CNN:
    def __init__(self, model_type = 'classifier', cmode = 'rgb', img_size = 64, size = 'medium',
                 loss_type = 'mean_squared_error', learn_rate = 0.001, final_act = 'sigmoid', opt_type = 'rms', dropout = True, eval_metric = 'accuracy'):
        self.model_type = model_type; self.learn_rate = learn_rate; self.loss_type = loss_type; self.histories = [];
        self.val_losses = []; self.val_accs = []; self.runtimes = []; self.cmode = cmode; self.img_size = img_size;
        self.trained = False; self.validated = False; self.scores = None; self.epochs = None; self.batch_size = None;
        self.save_name = None;self.final_act = final_act;self.dropout = dropout; self.opt_type = opt_type; self.models = []
        self.metric = eval_metric; self.size = size
        if self.cmode == 'rgb':
            self.channels = 3
        else:
            self.channels = 1
        return
    
    def train(self, x_train, y_train, x_test, y_test, epochs = 5, batch_size = 32, folds = 1):
        self.acc_list = []
        self.loss_list = []
        if self.model_type == 'classifier':
            if len(np.shape(y_train)) == 1:
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)
        print('\n'+'\t'.join([self.model_type, self.final_act, self.loss_type, str(self.epochs), self.opt_type, str(self.learn_rate)]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.trained = True
        self.folds = folds
        if self.folds <= 0:
            self.folds = 1
        for fold in range(self.folds):
            print('\n{} {}'.format(self.model_type.capitalize(), fold+1))
            print('------------------------------------------')

            st = time.time()
            if self.size == 'small':
                model = SmallCNN(self.img_size, model_type = self.model_type, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            if self.size == 'medium':
                model = MedCNN(self.img_size, model_type = self.model_type, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            if self.size == 'large':
                model = LargeCNN(self.img_size, model_type = self.model_type, channels = self.channels, opt_type = self.opt_type, losses = self.loss_type, final_act = self.final_act, learn_rate = self.learn_rate, dropout = self.dropout)
            self.models.append(model)
            self.histories.append(model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = False))
            if self.model_type == 'regressor':
                self.acc_list.append(self.histories[-1].history['val_acc'])
            else:
                self.acc_list.append(self.histories[-1].history['val_categorical_accuracy'])
            self.loss_list.append(self.histories[-1].history['loss'])
            dt = time.time() - st
            self.runtimes.append(dt)
        self.model = model
        if self.folds > 1:
            print('\n{}-Model Summary Stats:'.format(self.folds))
        else:
            print('Training Results:')
        
        print('------------------------------------------')
#        print('Accuracy:    {} +/- {}'.format(round(np.mean(self.acc_list),4), round(np.std(self.acc_list),2)))
#        print('Loss:        {} +/- {}'.format(round(np.mean(self.loss_list),4), round(np.std(self.loss_list),2)))
        print('Runtime:     {} +/- {}'.format(round(np.mean(self.runtimes),2), round(np.std(self.runtimes),2)))
        self.trained = True
        self.code = codifyModel(m_type = self.model_type, color_type = self.cmode, opt_type = self.opt_type,
                            final_act = self.final_act, lr = self.learn_rate)  
        return
    
    def evaluate(self, x_valid, y_valid, plots_on = True):
        if self.trained:
            for model in self.models:
                if self.model_type == 'classifier':
                    if len(np.shape(y_valid)) == 1:
                        y_valid = to_categorical(y_valid)
                i = 0
                mean_acc_list = np.zeros(self.epochs)
                mean_loss_list = np.zeros(self.epochs)
                
                for history in self.histories:
                    if self.model_type == 'regressor':
                        acc_list = history.history['val_acc']
                    else:
                        acc_list = history.history['val_categorical_accuracy']
    
                    loss_list = history.history['loss']
                    if i >= 1:
                        for j in range(self.epochs):
                            mean_acc_list[j] += acc_list[j]
                            mean_loss_list[j] += loss_list[j]
                    i+=1
                mean_acc_list = np.divide(mean_acc_list, len(mean_acc_list))
                mean_loss_list = np.divide(mean_loss_list, len(mean_loss_list))
                
                if plots_on:
                    plt.plot(mean_acc_list)
                    plt.plot(mean_loss_list)
                    plt.show()
                    for history in self.histories:
                        plt.figure(figsize = [7,4])
                        plt.plot(history.history['loss'],'r',linewidth = 2.0, linestyle = '--')
                        plt.plot(history.history['val_loss'],'b',linewidth = 2.0, linestyle = '--')
                        if self.model_type == 'classifier':
                            plt.plot(history.history['categorical_accuracy'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_categorical_accuracy'],'b',linewidth = 2.0)
                        elif self.model_type == 'regressor':
                            plt.plot(history.history['acc'],'r',linewidth = 2.0)
                            plt.plot(history.history['val_acc'],'b',linewidth = 2.0)
                        plt.legend(['Training Data', 'Test Data'], fontsize = 12)
                        plt.xlabel('Epochs', fontsize = 16)
                        plt.ylabel('Loss / Acc',fontsize = 16)
                        plt.title('{} {} {} ({}, {}, {})'.format(self.img_size, self.cmode.upper(), self.model_type.capitalize(), self.opt_type, self.final_act, self.learn_rate), fontsize = 16)
                        plt.show()
                self.scores = model.evaluate(x_valid, y_valid, verbose = 1)
                self.val_losses.append(self.scores[0])
                self.val_accs.append(self.scores[1])
            
            self.validated = True
            
            # Confusion Matrix
            if self.model_type == 'classifier':
                print('\nCLASSIFICATION ARCHITECTURE')
                print('------------------------------------------')
                y_valid_reduced = list(np.zeros(len(y_valid)))
                for i in range(len(y_valid)):
                    y_valid_reduced[i] = np.argmax(y_valid[i])
                self.val_acc_list = []
                self.prec_list = []
                self.spec_list = []
                self.sens_list = []
                self.cm_list = []
                for model in self.models:
                    cm = confusion_matrix(model.predict_classes(x_valid), y_valid_reduced)
                    tn, fp, fn, tp = cm.ravel()
                    model_acc = sum(cm.diagonal()) / cm.sum()
                    model_prec = tp / (tp + fp)
#                    model_spec = tn / (tn + fp)
                    model_sens = tp / (tp + fn)
                    self.val_acc_list.append(model_acc)
                    self.prec_list.append(model_prec)
#                    self.spec_list.append(model_spec)
                    self.sens_list.append(model_sens)
                    self.cm_list.append(cm)

                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))
                print('Precision   (PPV):  {} +/- {}'.format(round(np.mean(self.prec_list),4), round(np.std(self.prec_list),2)))
#                print('Specificity (TNR):  {} +/- {}'.format(round(np.mean(self.spec_list),4), round(np.std(self.spec_list),2)))
                print('Recall (TPR):       {} +/- {}'.format(round(np.mean(self.sens_list),4), round(np.std(self.sens_list),2)))

            # ROC & AUC Scores
            if self.model_type == 'regressor':
                print('\nREGRESSION ARCHITECTURE')
                print('------------------------------------------')
                self.auc_list = []
                for model in self.models:
                    fpr, tpr, threshold = roc_curve(y_valid, model.predict(x_valid))
                    auc_score = auc(fpr, tpr)
                    if plots_on:
                        plt.figure(figsize = (9,6))
                        plt.xlim(0, 0.2)
                        plt.ylim(0.8, 1)
                        plt.plot(fpr, tpr)
                        plt.plot([0, 1], [0, 1], 'k--', lw = 2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.xlim(left = 0, right = 1)
                        plt.ylim(top = 1, bottom = 0)
                    self.auc_list.append(auc_score)
                print('Accuracy:           {} +/- {}'.format(round(np.mean(self.val_accs),4), round(np.std(self.val_accs),2)))
                print('Loss:               {} +/- {}'.format(round(np.mean(self.val_losses),4), round(np.std(self.val_losses),2)))
                print('AUC:                {} +/- {}'.format(round(np.mean(self.auc_list),4), round(np.std(self.auc_list),2)))
        else:
            print('Model Not Trained!')
        return
    
    def load(self, fname = 'keras_model.h5'):
        self.models = []
        self.model = load_model(fname)
        print('Model loaded from: ', fname)
        return
    
    def save(self, fname = 'keras_model.h5'):
        self.model.save(fname)
        self.save_name = fname
        self.fsize = os.path.getsize(fname)
        print('Model saved to: {}\nSize:    {} MB'.format(fname, round(self.fsize/1024/1024,2)))
        return
    
    def plotFilters(self, img, img_class = None):
        if img_class != None:
            print(f'Class: {img_class}')
        plt.imshow(img)
        plt.show()
        print('\nConvolution Outputs')
        plotLayer(self.model, 0, img, normalize = False)
        plotLayer(self.model, 3, img, normalize = False)
        plotLayer(self.model, 6, img, normalize = False)
        return