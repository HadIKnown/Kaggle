# Initial exploration and coordinate retrieval from Kaggle kernel:
# https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
import os
import numpy as np
import pandas as pd

np.random.seed(42) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv3D, MaxPooling3D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import keras.backend as K
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split

def getXyzData(filename):
    pos_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                # there's a faster way to do this for sure
                if x[4] == 'O': atom = [1, 0, 0, 0]
                elif x[4] == 'Al': atom = [0, 1, 0, 0]
                elif x[4] == 'Ga': atom = [0, 0, 1, 0]
                elif x[4] == 'In': atom = [0, 0, 0, 1]
                pos_data.append([np.array(x[1:4], dtype = float), atom])
    return pos_data

def getAllXyzData(train_or_test):
    all_pos_data = []
    directories = os.listdir(train_or_test)
    directories.pop(0)
    directories = [int(directory) for directory in directories]
    directories.sort()
    for directory in directories:
        filename = train_or_test + '/' + str(directory) + '/geometry.xyz'
        all_pos_data.append(getXyzData(filename))
    return all_pos_data

data = getAllXyzData('train_geo')

# ANALYZE THE TEST DATA TO FIND MAX LENGTH THERE
# for now, 25 is max length (so 26 for 0 and 25)
def roundData(data):
    rounded_data = []
    #maxi = 0
    for sample in data:
        rounded_sample = []
        for atom in sample:
            rounded_atom = []
            rounded_atom = [np.round(atom[0]).astype(int), atom[1]]
            #if max(rounded_atom[0]) > maxi:
                #maxi = max(rounded_atom[0])
            rounded_sample.append(rounded_atom)
        rounded_data.append(rounded_sample)
    return rounded_data

rounded_data = roundData(data)

def getLatticeImage(data):
    lattices = []
    for sample in data:
        lattice = np.zeros((26, 26, 26, 4))
        for atom in sample:
            atom_coords = atom[0]
            x, y, z = (atom_coords[0], atom_coords[1], atom_coords[2])
            lattice[x, y, z] = atom[1]
        lattices.append(lattice)
    return lattices

lattice_imgs_train = np.array(getLatticeImage(rounded_data))
fe_bge_train = np.array(np.log1p(pd.read_csv('train.csv')[
                                            ['formation_energy_ev_natom', 
                                             'bandgap_energy_ev']]))

x_train, x_val, y_train, y_val = train_test_split(lattice_imgs_train,
                                                  fe_bge_train,
                                                  test_size = 0.01,
                                                  random_state = 42)
                                                  
                                                
########################################
### If predictions are much closer (like proper order of magnitude),
### submit predictions. THEN Add another conv, increase nodes everywhere, 
### then run on AWS again for the night. While running, set up TF GPU on 
### Tiensi PC.
#######################################

def getModel():
    #Build keras model
    
    model=Sequential()
    # Padding????/
    # CNN 1
    model.add(Conv3D(512, kernel_size=(5, 5, 5),activation='relu', #quadruple everyhting
                     input_shape=(26, 26, 26, 4)))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu' ))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu' ))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu')) #add the other features here
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output 
    model.add(Dense(2, activation='relu'))

    optimizer = SGD(lr=0.001)
    def rmsle(y_true, y_pred):
        return K.sqrt(mean_squared_error(y_true, y_pred))
    model.compile(loss=rmsle, optimizer=optimizer, metrics=[rmsle])
    
    return model

model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(x_val, y_val, verbose=1)
print('Train score:', score[0])

data_submit = np.array(getLatticeImage(roundData(getAllXyzData('test_geo'))))
pred_test = model.predict(data_submit)
print(pred_test)
pred_test = np.expm1(pred_test)
print(pred_test)

submission = pd.DataFrame({'id': pd.read_csv('test.csv')['id'], 
                           'formation_energy_ev_natom': 
                              pred_test[:,0].reshape((pred_test.shape[0])),
                           'bandgap_energy_ev': 
                              pred_test[:,1].reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('keras_submission.csv', index=False) 


