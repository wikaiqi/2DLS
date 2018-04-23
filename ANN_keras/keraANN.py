import numpy as np
import math
import tensorflow as tf
tf.set_random_seed(2)
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from keras.optimizers import Adam #, SGD
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from sklearn.model_selection import KFold #StratifiedKFold
import matplotlib.pylab as plt
from keras.callbacks import Callback

def NN_model(x, size, drop_rate, l2_parameter):
    '''
    Implement a three layers neural network using Keras
    Input:
       x            : input layer
       size         : number of features 
       drop_rate    : dropout rate
       l2_parameter : parameter for L2 regularizers
    return:
       y  : NN output layer 
    '''
    # layer 1
    A1 = Dense(25, input_shape=(size,), kernel_initializer="glorot_uniform", 
               kernel_regularizer=regularizers.l2(l2_parameter), activation="relu")(x)
    A1 = Dropout(drop_rate)(A1)
    
    #layer 2
    A2 = Dense(12, kernel_initializer="glorot_uniform", 
               kernel_regularizer=regularizers.l2(l2_parameter), activation="relu")(A1)
    A2 = Dropout(drop_rate)(A2)
    
    #output layer
    y  = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")(A2)

    return y

def Cross_Entropy(y_true, y_pred):
    '''
    Calculate binary cross entropy
    '''
    result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return np.mean(result)

def plot_train_history(train_loss, val_loss):
    plt.rcParams["figure.figsize"]=(8, 6)
    #fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()  
    
    
class Train_Cross_Entropy(Callback):
    def __init__(self, training_data=(), validation_data=(), interval=1):
        
        super(Callback, self).__init__()

        self.interval          = interval
        self.X    , self.y     = training_data
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred     = self.model.predict(self.X, verbose=0)
            score      = Cross_Entropy(self.y, y_pred)
            
            y_pred_val = self.model.predict(self.X_val, verbose=0)
            score_val  = Cross_Entropy(self.y_val, y_pred_val)
            
            print("  Epoch: {:3d} - Train_loss: {:.6f} - val_loss: {:.6f}".format(epoch, score, score_val))


#NN model
def run_ANN(train_x, train_y, test_x, test_y,  LJ_test_x_83, LJ_test_y_83, 
            LJ_test_x_85, LJ_test_y_85, LJ_test_x_87, LJ_test_y_87, LJ_test_x_89, LJ_test_y_89, 
            size, epochs, batch_size, lr_rate, droprate, L2, verbose=0, plot=0):

    skf = KFold(n_splits=5)
    loss_all           = []
    val_loss_all       = []
    train_entropy      = [] 
    test_entropy       = []
    train_pred_y       = []
    test_pred_y        = []
    LJ_test_entropy_83 = []
    LJ_test_pred_y_83  = []
    LJ_test_entropy_85 = []
    LJ_test_pred_y_85  = []
    LJ_test_entropy_87 = []
    LJ_test_pred_y_87  = []
    LJ_test_entropy_89 = []
    LJ_test_pred_y_89  = []
    for train_index, val_index in skf.split(train_x, train_y):
        
        X_train, X_val = train_x[train_index], train_x[val_index]
        y_train, y_val = train_y[train_index], train_y[val_index]
    
        x = Input(shape=(size,))
        y = NN_model(x, size, droprate, L2)
        model = Model(inputs=x, outputs=y)
        #model.summary() 
    
        adam = Adam(lr=lr_rate, decay=0.0, amsgrad=True)
        #sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        #mcp_save       = ModelCheckpoint('ANN.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
        ra_val = Train_Cross_Entropy(training_data =(X_train, y_train), 
                                  validation_data=(X_val, y_val), interval=1)
    
        # Build a model with a n_h-dimensional hidden layer
        history=model.fit(X_train, y_train, batch_size=batch_size, 
                      epochs=epochs, verbose=verbose,  validation_data=(X_val, y_val), callbacks=[ra_val], shuffle=False)
        loss_all.append(history.history['loss'])
        val_loss_all.append(history.history['val_loss'])
    
        #model.load_weights(filepath = 'ANN.hdf5')
        #Evaluate on training set
        train_pred_y_cv  = model.predict(train_x, verbose=verbose)
        train_entropy_cv = Cross_Entropy(train_y, train_pred_y_cv)
        train_entropy.append(train_entropy_cv)
        train_pred_y.append(train_pred_y_cv)
    
        #Evaluate on test set
        test_pred_y_cv  = model.predict(test_x, verbose=verbose)
        test_entropy_cv = Cross_Entropy(test_y, test_pred_y_cv)
        test_entropy.append(test_entropy_cv)
        test_pred_y.append(test_pred_y_cv)
        
        #Evaluate on LJ test set
        LJ_test_pred_y_cv_83  = model.predict(LJ_test_x_83, verbose=verbose)
        LJ_test_entropy_cv_83 = Cross_Entropy(LJ_test_y_83, LJ_test_pred_y_cv_83)
        LJ_test_entropy_83.append(LJ_test_entropy_cv_83)
        LJ_test_pred_y_83.append(LJ_test_pred_y_cv_83)
        
        LJ_test_pred_y_cv_85  = model.predict(LJ_test_x_85, verbose=verbose)
        LJ_test_entropy_cv_85 = Cross_Entropy(LJ_test_y_85, LJ_test_pred_y_cv_83)
        LJ_test_entropy_85.append(LJ_test_entropy_cv_85)
        LJ_test_pred_y_85.append(LJ_test_pred_y_cv_85)
        
        LJ_test_pred_y_cv_87  = model.predict(LJ_test_x_87, verbose=verbose)
        LJ_test_entropy_cv_87 = Cross_Entropy(LJ_test_y_87, LJ_test_pred_y_cv_83)
        LJ_test_entropy_87.append(LJ_test_entropy_cv_87)
        LJ_test_pred_y_87.append(LJ_test_pred_y_cv_87)
        
        LJ_test_pred_y_cv_89  = model.predict(LJ_test_x_89, verbose=verbose)
        LJ_test_entropy_cv_89 = Cross_Entropy(LJ_test_y_89, LJ_test_pred_y_cv_83)
        LJ_test_entropy_89.append(LJ_test_entropy_cv_89)
        LJ_test_pred_y_89.append(LJ_test_pred_y_cv_89)
        
        print('-------------------------------------------------------')

    train_loss = np.mean(loss_all    , axis=0)
    val_loss   = np.mean(val_loss_all, axis=0)
    
    train_entropy_mean = np.mean(train_entropy, axis=0)
    train_entropy_std  = np.std(train_entropy, axis=0)
    
    test_entropy_mean  = np.mean(test_entropy,  axis=0)
    test_entropy_std   = np.std(test_entropy,  axis=0)
    
    entropy = [train_entropy_mean, train_entropy_std, test_entropy_mean, test_entropy_std]
    
    if plot:
        plot_train_history(train_loss, val_loss)
    
    return train_loss, val_loss, test_pred_y, LJ_test_pred_y_83, LJ_test_pred_y_85, LJ_test_pred_y_87,LJ_test_pred_y_89, entropy
