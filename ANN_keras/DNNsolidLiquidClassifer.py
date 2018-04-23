# coding: utf-8
# Classify Solid and Liquid phases with Neural Network
# 
# Author: Weikai Qi
# Email : wikaiqi@gmail.com
# 
# In this project, I will train a neural network to regonize solid 
# and liquid phases in two-dimension from configurations generating in MD simulation. 

# In[1]:
import argparse
# Setting parameters
parser = argparse.ArgumentParser(description='ANN for solid-liquid classifier')
parser.add_argument('-d', '--droprate', type=float, default=0.5,   help='Dropout rate')
parser.add_argument('-lr', '--lr_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-L2', '--L2_nrom', type=float, default=0.05,  help='L2 nrom parameters for weights')
parser.add_argument('-b',  '--batch',   type=int,   default=128,   help='Batch size')
parser.add_argument('-e',  '--epochs',  type=int,   default=50,    help='Epochs')
parser.add_argument('-m',  '--t_mode',  type=int,   default=0,     help='test mode = 0/1: Exclude/Include test samples at packf [0.68-0.74] (coexistance phase) ')
arg        = parser.parse_args()

import numpy      as np
np.random.seed(1)
from loaddata import load_HDdata, load_LJdata
from keraANN  import run_ANN
import pandas as pd
import time

# In[2]:
start = time.clock()

droprate   = arg.droprate  # dropout rate
L2         = arg.L2_nrom   # L2 nrom parameters for weights
lr_rate    = arg.lr_rate   # learning rate
batch_size = arg.batch     # batch szie
epochs     = arg.epochs    # epochs
t_mode     = arg.t_mode    # test sample mode

yesno= lambda x: 'yes' if x else 'no'

print('-------------------------------------------------------')
print('Dropout rate              : {}'.format(droprate))
print('Learning rate             : {}'.format(lr_rate))
print('L2 nrom paramters         : {}'.format(L2))
print('Batch size                : {}'.format(batch_size))
print('number of epochs          : {}'.format(epochs))
print('Include coexistane        : {}'.format(yesno(t_mode)))
print('-------------------------------------------------------')

#load data
HD_train_solid_path  = '../data/HardDiskTrainsetSolid.csv'
HD_train_liquid_path = '../data/HardDiskTrainsetLiquid.csv'
HD_test_path         = '../data/HardDiskTestset.csv'
LJ_test_path_87      = '../data/LJ0873Testset.csv'
LJ_test_path_83      = '../data/LJ0830Testset.csv'
LJ_test_path_85      = '../data/LJ0850Testset.csv'
LJ_test_path_89      = '../data/LJ0890Testset.csv'

HD_train_x, HD_train_y, HD_test_x, HD_test_y, HD_test_packf, n_test_packf, size = load_HDdata(
                                                                    HD_train_solid_path, HD_train_liquid_path, 
                                                                    HD_test_path,
                                                                    shuffle=1, verbose=1, mode=t_mode) 

LJ_test_x_87, LJ_test_y_87, LJ_test_T_87, n_test_T_87 = load_LJdata(LJ_test_path_87, verbose=1)
LJ_test_x_83, LJ_test_y_83, LJ_test_T_83, n_test_T_83 = load_LJdata(LJ_test_path_83, verbose=1)
LJ_test_x_85, LJ_test_y_85, LJ_test_T_85, n_test_T_85 = load_LJdata(LJ_test_path_85, verbose=1)
LJ_test_x_89, LJ_test_y_89, LJ_test_T_89, n_test_T_89 = load_LJdata(LJ_test_path_89, verbose=1)


# train ANN
train_loss, val_loss, test_pred_y, LJ_test_pred_y_83, LJ_test_pred_y_85,LJ_test_pred_y_87,LJ_test_pred_y_89,entropy = run_ANN(HD_train_x, HD_train_y, 
                                                     HD_test_x, HD_test_y, 
                                                     LJ_test_x_83, LJ_test_y_83,
                                                     LJ_test_x_85, LJ_test_y_85,
                                                     LJ_test_x_87, LJ_test_y_87,
                                                     LJ_test_x_89, LJ_test_y_89,
                                                     size, epochs, batch_size, lr_rate, 
                                                     droprate, L2, verbose=0, plot=0)
 
# In[3]:
# ## Model evaluation
elapsed_time_sec   = (time.clock() - start)

print('cross entropy on train set: {:.6f} +/- {:6f}'.format(entropy[0], entropy[1]))
print('cross entropy on test  set: {:.6f} +/- {:6f}'.format(entropy[2], entropy[3]))
print('Elapsed time (seconds)    : {:.4f}'.format(elapsed_time_sec))
print('Elapsed time (mins)       : {:.4f}'.format(elapsed_time_sec/60))
print('--------------------------------------------------------')

#Save results to file
n_each_packf        = int(HD_test_packf.shape[0]/ n_test_packf)
test_pred_y_mean    = np.mean(test_pred_y, axis=0)
test_pred_y_std     = np.std(test_pred_y,  axis=0)

pack      = np.mean(HD_test_packf.reshape(n_test_packf, n_each_packf),    axis=1)
test_pred = np.mean(test_pred_y_mean.reshape(n_test_packf, n_each_packf), axis=1)
test_std  = np.mean(test_pred_y_std.reshape(n_test_packf, n_each_packf),  axis=1)

pred = pd.DataFrame({'packf':pack, 'test_pred':test_pred, 'test_std': test_std})
pred.to_csv('NN_pred_on_test.csv')

LJ_test_pred_y_mean = np.mean(LJ_test_pred_y_83, axis=0)
LJ_test_pred_y_std   = np.std(LJ_test_pred_y_83, axis=0)

n_each_T        = int(LJ_test_T_83.shape[0]/ n_test_T_83)
T_LJ         = np.mean(LJ_test_T_83.reshape(n_test_T_83, n_each_T),        axis=1)
LJ_test_pred = np.mean(LJ_test_pred_y_mean.reshape(n_test_T_83, n_each_T), axis=1)
LJ_test_std  = np.mean(LJ_test_pred_y_std.reshape(n_test_T_83, n_each_T),  axis=1)

pred = pd.DataFrame({'T':T_LJ, 'test_pred':LJ_test_pred, 'test_std': LJ_test_std})
pred.to_csv('NN_LJ_pred_on_test_83.csv')

LJ_test_pred_y_mean = np.mean(LJ_test_pred_y_85, axis=0)
LJ_test_pred_y_std   = np.std(LJ_test_pred_y_85, axis=0)

n_each_T        = int(LJ_test_T_85.shape[0]/ n_test_T_85)
T_LJ         = np.mean(LJ_test_T_85.reshape(n_test_T_85, n_each_T),        axis=1)
LJ_test_pred = np.mean(LJ_test_pred_y_mean.reshape(n_test_T_85, n_each_T), axis=1)
LJ_test_std  = np.mean(LJ_test_pred_y_std.reshape(n_test_T_85, n_each_T),  axis=1)

pred = pd.DataFrame({'T':T_LJ, 'test_pred':LJ_test_pred, 'test_std': LJ_test_std})
pred.to_csv('NN_LJ_pred_on_test_85.csv')

LJ_test_pred_y_mean = np.mean(LJ_test_pred_y_87, axis=0)
LJ_test_pred_y_std   = np.std(LJ_test_pred_y_87, axis=0)

n_each_T        = int(LJ_test_T_87.shape[0]/ n_test_T_87)
T_LJ         = np.mean(LJ_test_T_87.reshape(n_test_T_87, n_each_T),        axis=1)
LJ_test_pred = np.mean(LJ_test_pred_y_mean.reshape(n_test_T_87, n_each_T), axis=1)
LJ_test_std  = np.mean(LJ_test_pred_y_std.reshape(n_test_T_87, n_each_T),  axis=1)

pred = pd.DataFrame({'T':T_LJ, 'test_pred':LJ_test_pred, 'test_std': LJ_test_std})
pred.to_csv('NN_LJ_pred_on_test_87.csv')

LJ_test_pred_y_mean = np.mean(LJ_test_pred_y_89, axis=0)
LJ_test_pred_y_std   = np.std(LJ_test_pred_y_89, axis=0)

n_each_T        = int(LJ_test_T_89.shape[0]/ n_test_T_89)
T_LJ         = np.mean(LJ_test_T_89.reshape(n_test_T_89, n_each_T),        axis=1)
LJ_test_pred = np.mean(LJ_test_pred_y_mean.reshape(n_test_T_89, n_each_T), axis=1)
LJ_test_std  = np.mean(LJ_test_pred_y_std.reshape(n_test_T_89, n_each_T),  axis=1)

pred = pd.DataFrame({'T':T_LJ, 'test_pred':LJ_test_pred, 'test_std': LJ_test_std})
pred.to_csv('NN_LJ_pred_on_test_89.csv')



