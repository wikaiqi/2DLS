import pandas as pd
import numpy as np
import time
#load data

def load_HDdata(train_solid_file, train_liquid_file, test_file, shuffle=0, verbose=1, mode=0):
    '''
    load hard disk training and test data
    Input: 
        train_solid_file  : path to solid phase training dataset
        train_liquid_file : path to liquid phase training dataset
        test_file         : path to test dataset
        shuffle           : 1/shuffle the training set; 0/ no shuffle
        verbose           : 1/print out information; 0/ not print any information
    Return:
        HD_train_x  : features in training set [(x, y) coordinates]
        HD_train_y  : labels in training set
        HD_test_x   : features in test set
        HD_test_y   : labels in test set
    '''
    HD_train_solid  = pd.read_csv(train_solid_file)
    HD_train_liquid = pd.read_csv(train_liquid_file)
    HD_test         = pd.read_csv(test_file)
    HD_train        = pd.concat([HD_train_liquid, HD_train_solid], axis = 0) 
    
    Train_data = HD_train.drop(['id','packf'], axis=1).values

    # shuffle dataSet
    if shuffle:
        np.random.seed(1)
        np.random.shuffle(Train_data)
    
    size = HD_train.shape[1] - 3
    HD_train_x = Train_data[:, 0: size]
    HD_train_y = Train_data[:, -1]
    HD_train_y = HD_train_y.reshape(HD_train_y.shape[0], 1)
    
    #print information about test dataset
    if mode==0:
        HD_test_liquid = HD_test[HD_test['packf']<0.69]
        HD_test_solid  = HD_test[HD_test['packf']>0.73]
    else:
        HD_test_liquid = HD_test[HD_test['packf']<0.70]
        HD_test_solid  = HD_test[HD_test['packf']>0.69]
    HD_test_ex     = pd.concat([HD_test_liquid, HD_test_solid], axis=0)
    
    n_test_packf   = HD_test_ex['packf'].unique().shape[0]
    n_train_packf  = HD_train['packf'].unique().shape[0]
    
    HD_test_y      = HD_test_ex['type'].values
    HD_test_y      = HD_test_y.reshape(HD_test_y.shape[0], 1)
    HD_test_x      = HD_test_ex.drop(['type','id','packf'], axis=1).values
    HD_test_packf  = HD_test_ex['packf'].values
    
    if verbose:
        #print information about training dataset
        print("number of liquid training samples : ", HD_train_solid.shape[0])
        print("number of solid  training samples : ", HD_train_liquid.shape[0])
        print("number of test            samples : ", HD_test.shape[0])
        print("number of                features : ", size)
        print("number of packing fraction (train): ", n_train_packf)
        print("number of packing fraction (test) : ", n_test_packf)
        print("HD_train_x  shape                 : ", HD_train_x.shape)
        print("HD_train_y  shape                 : ", HD_train_y.shape)
        print("HD_test_x data shape              : ", HD_test_x.shape)
        print("HD_test_y data shape              : ", HD_test_y.shape)
        print('-------------------------------------------------------')
    
    return HD_train_x, HD_train_y, HD_test_x, HD_test_y, HD_test_packf, n_test_packf, size

if __name__=='__main__':
    
    start = time.clock()
    
    HD_train_solid_path  = 'data/HardDiskTrainsetSolid.csv'
    HD_train_liquid_path = 'data/HardDiskTrainsetLiquid.csv'
    HD_test_path         = 'data/HardDiskTestset.csv'

    train_x, train_y, test_x, test_y, _ = load_HDdata(HD_train_solid_path, 
                                                   HD_train_liquid_path, HD_test_path,
                                                   shuffle=1, verbose=1) 
    
    elapsed_time_sec   = (time.clock() - start)
    
    print('Elapsed time: {:10.4f} second'.format(elapsed_time_sec))
    