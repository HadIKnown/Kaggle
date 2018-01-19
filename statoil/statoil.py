# layer syntax found here: https://www.tensorflow.org/tutorials/
# layers#convolutional_layer_1

#Try to change density of FC layers in the morning

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.set_random_seed(42) #doesn't work
tf.logging.set_verbosity(tf.logging.INFO)

#was getting shape error for input, but set num_epochs = 1 and it wnt away...?
def get_input_fn(X, y, batch_size = 24, num_epochs = 200, shuffle = True):
    return tf.estimator.inputs.numpy_input_fn(
             x = {'x': X},
             y = y,
             num_epochs = num_epochs,
             shuffle = shuffle) 

def my_model(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 75, 75, 2]) #might need to specify features more...
    conv1 = tf.layers.conv2d(
              inputs = input_layer,
              filters = 20,
              kernel_size = 10,
              padding = 'same',
              activation = tf.nn.relu,
              kernel_initializer = None)
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 3, strides = 3)
    conv2 = tf.layers.conv2d(
              inputs = pool1,
              filters = 40,
              kernel_size = 10,
              padding = 'same',
              activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 5, strides = 5)
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 40]) #calculated independently
    dense = tf.layers.dense(inputs = pool2_flat, units = 100) #maybe more units
    dropout = tf.layers.dropout(inputs = dense,
                rate = 0.5, 
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs = dropout, units = 2)
    
    predictions = {
      'classes': tf.argmax(input = logits, axis = 1),
      'probabilities': tf.nn.softmax(logits, name = 'softmax_tensor')
    }
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode,
          predictions = predictions)

    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 2)
    loss = tf.losses.log_loss(predictions = 
                                predictions['probabilities'],
                              labels = onehot_labels)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss,
                                      global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss,
                                          train_op = train_op)

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(
                                       labels = labels,
                                       predictions = predictions['classes'])}
    return tf.estimator.EstimatorSpec(
               mode = mode,
               loss = loss,
               eval_metric_ops = eval_metric_ops)
                                                   
def main(unused_argv, use_angles = False):
    raw = pd.read_json('data-1/processed/train.json')
    raw.pop('id')
    if use_angles:
        inc_angle_na = []
        for i in range(0, raw.shape[0]):
            if raw['inc_angle'].iloc[i] == 'na': inc_angle_na.append(i) 
        raw.drop(labels = inc_angle_na, axis = 0, inplace = True)
    else: raw.pop('inc_angle')
    X_band_1 = np.array([np.array(band).astype(np.float32) 
                         for band in raw['band_1']])
    X_band_2 = np.array([np.array(band).astype(np.float32) 
                         for band in raw['band_2']])
    X = np.concatenate([X_band_1[:, :, np.newaxis], X_band_2[:, :, np.newaxis]],
                        axis = -1)
    y = np.array(raw['is_iceberg'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                        train_size = 0.8)
    
    raw_predict = pd.read_json('data/processed/test.json')
    pred_id = pd.DataFrame(raw_predict.pop('id'))
    if use_angles:
        inc_angle_na = []
        for i in range(0, raw_predict.shape[0]):
            if raw_predict['inc_angle'].iloc[i] == 'na': inc_angle_na.append(i) 
        raw_predict.drop(labels = inc_angle_na, axis = 0, inplace = True)
    else: raw_predict.pop('inc_angle')
    X_band_1 = np.array([np.array(band).astype(np.float32) 
                         for band in raw_predict['band_1']])
    X_band_2 = np.array([np.array(band).astype(np.float32) 
                         for band in raw_predict['band_2']])
    X_predict = np.concatenate([X_band_1[:, :, np.newaxis], X_band_2[:, :, np.newaxis]],
                        axis = -1)
    """
    #could cause issues if I need lists instead of tuples
    bands = []
    for i in range(raw.shape[0]):
        #bands.append(list(zip(band_1[i], band_2[i])))
    data = pd.concat([raw, pd.DataFrame(bands)], axis = 1, join = 'inner') 
    """
    #could just use train test split
    #testing = data.iloc[int(0.8 * len(data)):]
    #training = data.iloc[:int(0.8 * len(data))]
    
    classifier = tf.estimator.Estimator(model_fn = my_model,
                                        model_dir = '/tmp/statoil_logloss1')
    
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                                              every_n_iter = 50)

    classifier.train(input_fn = get_input_fn(X_train, y_train),
                     hooks = [logging_hook])
    
    eval_results = classifier.evaluate(input_fn =
                                         get_input_fn(X_test, y_test,
                                                     batch_size = None,
                                                     num_epochs = 1,
                                                     shuffle = False))
    print(eval_results)
 
    y_predict = classifier.predict(input_fn =
                                       get_input_fn(
                                           X_predict, 
                                           batch_size = None,
                                           y = None,
                                           num_epochs = 1,
                                           shuffle = False))
    y_predict_submission = pd.DataFrame(list(p['classes'] 
                                        for p in itertools.islice(
                                         y_predict, raw_predict.shape[0])),
                                        columns = 'is_iceberg') 
    submission = pd.concat([predict_id, y_predict_submission], axis = 1, join = 'inner')
    submission.to_csv('cnn_submission.csv', index = False)
    #tf_features = []
    #for feat in FEATURES: tf_features.append(
     #                       tf.feature_column.numeric_column(feat))
    
if __name__ == "__main__":
  tf.app.run()  
