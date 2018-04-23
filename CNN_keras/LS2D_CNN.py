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
parser = argparse.ArgumentParser(description='CNN for solid-liquid classifier')
parser.add_argument('-d', '--droprate', type=float, default=0.5,   help='Dropout rate')
parser.add_argument('-lr', '--lr_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-L2', '--L2_nrom', type=float, default=0.05,  help='L2 nrom parameters for weights')
parser.add_argument('-b',  '--batch',   type=int,   default=128,   help='Batch size')
parser.add_argument('-e',  '--epochs',  type=int,   default=5,    help='Epochs')
parser.add_argument('-m',  '--t_mode',  type=int,   default=0,     help='test mode = 0/1: Exclude/Include test samples at packf [0.68-0.74] (coexistance phase) ')
arg        = parser.parse_args()

import numpy      as np
np.random.seed(1)
from loaddata import load_HDdata
from keraCNN  import run_CNN
from reshapedata import reshapeData
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

HD_train_x, HD_train_y, HD_test_x, HD_test_y, HD_test_packf, n_test_packf, size = load_HDdata(
                                                                    HD_train_solid_path, HD_train_liquid_path, 
                                                                    HD_test_path,
                                                                    shuffle=1, verbose=1, mode=t_mode) 

# In[2]
HD_train_x = reshapeData(HD_train_x)
HD_test_x  = reshapeData(HD_test_x)


# In[2]

# train ANN
train_loss, val_loss, test_pred_y, entropy = run_CNN(HD_train_x, HD_train_y, 
                                                     HD_test_x, HD_test_y, 
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
n_each_packf     = int(HD_test_packf.shape[0]/ n_test_packf)
test_pred_y_mean = np.mean(test_pred_y, axis=0)
test_pred_y_std  = np.std(test_pred_y,  axis=0)

pack      = np.mean(HD_test_packf.reshape(n_test_packf, n_each_packf),    axis=1)
test_pred = np.mean(test_pred_y_mean.reshape(n_test_packf, n_each_packf), axis=1)
test_std  = np.mean(test_pred_y_std.reshape(n_test_packf, n_each_packf),  axis=1)

pred = pd.DataFrame({'packf':pack, 'test_pred':test_pred, 'test_std': test_std})
pred.to_csv('NN_pred_on_test.csv')

