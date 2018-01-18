# layer syntax found here: https://www.tensorflow.org/tutorials/
# layers#convolutional_layer_1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def get_input_fn(data, num_epochs = None, shuffle = True):
    return tf.estimator.inputs.pandas_input_fn(
             x = pd.DataFrame({feat: data[feat].values for feat in list(data)}),
             y = pd.Series(data['is_iceberg'].values),
             num_epochs = num_epochs,
             shuffle = shuffle) 

def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    #play with specifying the channel number (which I think is the depth of the layr)
    

def main(unused_argv, use_angles = False):
    raw = pd.read_json('data-1/processed/train.json')
    raw.pop('id')
    if use_angles:
        inc_angle_na = []
        for i in range(0, raw.shape[0]):
            if raw['inc_angle'].iloc[i] == 'na': inc_angle_na.append(i) 
        raw.drop(labels = inc_angle_na, axis = 0, inplace = True)
    else: raw.pop('inc_angle')
    #investigate why the following line works!
    """get_intensities = pd.DataFrame(raw['band_1'].values.tolist())
    band_columns = {i: ('band_1_' + str(i)) 
                         for i in range(0, get_intensities.shape[1])}
    get_intensities.rename(columns = band_columns, inplace = True)

    get_intensities_2 = pd.DataFrame(raw['band_2'].values.tolist())
    band_columns_2 = {i: ('band_2_' + str(i)) 
                           for i in range(0, get_intensities.shape[1])}
    get_intensities_2.rename(columns = band_columns_2, inplace = True)

    raw.drop(['band_1', 'band_2'], axis = 1, inplace = True)
    data = pd.concat([raw, get_intensities, get_intensities_2],
                    axis = 1, join = 'inner')"""
    #could cause issues if I need lists instead of tuples
    #also lazy to add column to raws... but I guess... whatever
    bands = []
    band_1 = raw.pop('band_1')
    band_2 = raw.pop('band_2')
    for i in range(raw.shape[0]):
        bands.append(list(zip(band_1[i], band_2[i])))
    data = pd.concat([raw, pd.DataFrame(bands)], axis = 1, join = 'inner') 

    #could just use train test split
    testing = data.iloc[int(0.8 * len(data)):]
    training = data.iloc[:int(0.8 * len(data))]
  
    FEATURES = list(data)
    LABEL = 'is_iceberg'

    tf_features = []
    for feat in FEATURES: tf_features.append(
                            tf.feature_column.numeric_column(feat))
    
if __name__ == "__main__":
  tf.app.run()  
