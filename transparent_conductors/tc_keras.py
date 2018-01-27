# Initial exploration and coordinate retrieval from Kaggle kernel:
# https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
import os
import numpy as np
import pandas as pd

np.random.seed(42) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

negative_adjust = 2
#Holy crap okay 0 indexing makes the world tough.
#I want img_size to be 26, so that I have 0 and 25 as positions for atoms.
#That leaves me with an even number of positions, but I need a center.
#So add +1 to get to 27. Now I have positions 0 to 26.
# I want the 14th position to be the center, but that's the 13th index!
img_size = 26 + 1 + negative_adjust  #max length of atom position rounded plus 1, then add 1 if even

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
                # maybe do atomic properties instead
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

def getLatticeImage(data, rotate = False, transform_mat = np.array([])): #maybe add []
    # should take in argument that sets whether the last dimension is 1 or 4
    lattices = []
    for sample in data:
        lattice = np.zeros((img_size, img_size, img_size, 4))
        for atom in sample:
            atom_coords = atom[0]
            x, y, z = (atom_coords[0] + negative_adjust, #due to negative values ~ -2 
                       atom_coords[1] + negative_adjust, 
                       atom_coords[2] + negative_adjust) 
            if not rotate:
                lattice[x, y, z] = atom[1]
            else:
                new_coords =  np.matmul(transform_mat, 
                                        np.array([x, y, z, 1])
                                       ).astype(int)
                x_new, y_new, z_new = (new_coords[0], 
                                       new_coords[1], 
                                       new_coords[2])
                lattice[x_new, y_new, z_new] = atom[1]
        lattices.append(lattice)
    return np.array(lattices)

lattice_imgs_train = getLatticeImage(rounded_data)
csv_data_train = pd.read_csv('train.csv')
fe_bge_train = (np.log1p(csv_data_train[['formation_energy_ev_natom', 
                                                 'bandgap_energy_ev']]))
csv_data_train.drop(['formation_energy_ev_natom', 'bandgap_energy_ev', 'id'],
                     axis = 1, #consider dropping lattice parameters
                     inplace = True)
#Change the CNN to use Functional API to add in the extra features

# GET THIS WORKING IN THE MORNING. Rotate around X, Y, Z. That will 10x the data set.

def getMoreImgs(data):
    ### Check which spacegroups shouldn't be rotated due to repeats in symmetry.
    ### Will cause issues in matching up the fe_bge_train data if there is symmetry
    # Formulas from: http://www.math.tau.ac.il/~dcor/Graphics/cg-slides/geom3d.pdf
    # and: https://math.stackexchange.com/questions/1676441/how-to-rotate-the-
    # positions-of-a-matrix-by-90-degrees
    trans_zs = []
    trans_ys = []
    trans_xs = []

    T = np.array([[1, 0, 0, -(img_size - 1)/2],
                  [0, 1, 0, -(img_size - 1)/2],
                  [0, 0, 1, -(img_size - 1)/2],
                  [0, 0, 0, 1]])
    T_inv = np.linalg.inv(T) 
    for angle in [np.pi/2, np.pi, 3*np.pi/2]: 
        Rz = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                             [np.sin(angle), np.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]) 
        Ry = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                             [0, 1, 0, 0],
                             [-np.sin(angle), 0, np.cos(angle), 0],
                             [0, 0, 0, 1]])
        Rx = np.array([[1, 0, 0, 0],
                             [0, np.cos(angle), -np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0, 0, 0, 1]])
        RzT = np.matmul(Rz, T)
        RyT = np.matmul(Ry, T)
        RxT = np.matmul(Rx, T)
        trans_zs.append(np.matmul(T_inv, RzT))
        trans_ys.append(np.matmul(T_inv, RyT))
        trans_xs.append(np.matmul(T_inv, RxT))

    rot_90_x = getLatticeImage(data, rotate = True, 
                               transform_mat = trans_xs[0])
    rot_180_x = getLatticeImage(data, rotate = True, 
                               transform_mat = trans_xs[1])
    rot_270_x = getLatticeImage(data, rotate = True, 
                               transform_mat = trans_xs[2])
    rot_90_y = getLatticeImage(data, rotate = True, 
                               transform_mat = trans_ys[0])
    rot_180_y = getLatticeImage(data, rotate = True, 
                                transform_mat = trans_ys[1])
    rot_270_y = getLatticeImage(data, rotate = True, 
                                transform_mat = trans_ys[2]) 
    rot_90_z = getLatticeImage(data, rotate = True, 
                               transform_mat = trans_zs[0])
    rot_180_z = getLatticeImage(data, rotate = True, 
                                transform_mat = trans_zs[1])
    rot_270_z = getLatticeImage(data, rotate = True, 
                                transform_mat = trans_zs[2])  
    return np.concatenate((rot_90_x, rot_180_x, rot_270_x, 
                           rot_90_y, rot_180_y, rot_270_y,
                           rot_90_z, rot_180_z, rot_270_z))


more_lattice_imgs_train = np.concatenate(
                            (lattice_imgs_train,
                             getMoreImgs(rounded_data)))
more_fe_bge_train = np.concatenate((fe_bge_train, fe_bge_train,
                                    fe_bge_train, fe_bge_train,
                                    fe_bge_train, fe_bge_train,
                                    fe_bge_train, fe_bge_train,
                                    fe_bge_train, fe_bge_train))

"""
x_train, x_val, y_train, y_val = train_test_split(lattice_imgs_train,
                                                  fe_bge_train,
                                                  test_size = 0.01,
                                                  random_state = 42)
"""                                               
                                                
########################################
### If predictions are much closer (like proper order of magnitude),
### submit predictions. THEN Add another conv, increase nodes everywhere, 
### then run on AWS again for the night. While running, set up TF GPU on 
### Tiensi PC. [12, 33, 167, 194, 206, 227] Check symmetry.
#######################################

def getModel():
    #Build keras model
    
    model=Sequential()
    # Padding????/
    # CNN 1
    model.add(Conv3D(128, kernel_size=(6, 6, 6),activation='relu', #quadruple everyhting
                     input_shape=(img_size, img_size, img_size, 4)))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu' ))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    #model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu' ))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    #model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(256, activation='relu')) #add the other features here
    model.add(Dropout(0.2))

    #Dense 2
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    
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
#CHANGED SHUFFLE TO FALSE TO FEED IN THE EXTRA PARAMETERS. There's probably a better way.
model.fit(more_lattice_imgs_train, more_fe_bge_train, batch_size=batch_size, epochs=1, verbose=1, shuffle = False, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(lattice_imgs_train, fe_bge_train, verbose=1)
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


