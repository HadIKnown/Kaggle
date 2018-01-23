# This was used for the Statoil Iceberg Classifier Kaggle competition.
# By the end of the competition, the final log loss score I received was 0.4312.
# I had never written a CNN before nor done an independent project on image 
# classification. My primary struggle, however, resulted from outputting the
# predicted class (iceberg or not) instead of the raw probabilities. With
# more time, and a bit more attention to the exact submission requirements,
# I'm confident I would've raised the score further.
#
# In the future, I'd like to avoid the high level APIs and write a similar
# NN using only TF Core.
#
# Much of the structure for this script came from: https://www.tensorflow.org/
# tutorials/layers#convolutional_layer_1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

# Create an input function that feeds into the train, evaluate, and predict
# modes of the tf.Estimator
def get_input_fn(X, y, batch_size = 24, num_epochs = 30, shuffle = True):
    return tf.estimator.inputs.numpy_input_fn(
             x = {'x': X},
             y = y,
             batch_size = 24,
             num_epochs = num_epochs,
             shuffle = shuffle) 

# The architecture of the model is established here.
def my_model(features, labels, mode):
    # Architecture
    input_layer = tf.reshape(features['x'], [-1, 75, 75, 3])
    conv1 = tf.layers.conv2d(
              inputs = input_layer,
              filters = 64,
              kernel_size = 3,
              padding = 'same',
              activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 3, strides = 2)
    dropout1 = tf.layers.dropout(inputs = pool1,
                rate = 0.2,
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    conv2 = tf.layers.conv2d(
              inputs = dropout1,
              filters = 128,
              kernel_size = 3,
              padding = 'same',
              activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 3, strides = 2)
    dropout2 = tf.layers.dropout(inputs = pool2,
                rate = 0.2, 
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    conv3 = tf.layers.conv2d(
              inputs = dropout2,
              filters = 256,
              kernel_size = 3,
              padding = 'same',
              activation = tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = 2, strides = 2)
    dropout3 = tf.layers.dropout(inputs = pool3,
                rate = 0.2, 
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    # I'd imagine there's an automatic way to calculate the length of
    # the flattened dropout, but it was simple enough to calculate.
    dropout3_flat = tf.reshape(dropout3, [-1, 9 * 9 * 256])
    dense1 = tf.layers.dense(inputs = dropout3_flat, 
                             units = 1024,
                             activation = tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs = dense1,
                rate = 0.5, 
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    dense2 = tf.layers.dense(inputs = dropout4, 
                             units = 512,
                             activation = tf.nn.relu)
    dropout5 = tf.layers.dropout(inputs = dense2,
                rate = 0.3,
                training = (mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs = dropout5, units = 2)
    
    # Predictions
    predictions = {
      'classes': tf.argmax(input = logits, axis = 1),
      'probabilities': tf.nn.sigmoid(logits, name = 'sigmoid_tensor')
    }
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode,
          predictions = predictions)

    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 2)
    loss = tf.losses.sigmoid_cross_entropy(logits = logits,
                              multi_class_labels = onehot_labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
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
              
def get_more_images(imgs):
    # Originally found in: https://www.kaggle.com/cbryant/keras-cnn-statoil
    # -iceberg-lb-0-1995-now-0-1516
    """
    augmentation for more data
    """    


    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
 
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)        
    
    #Dropped vertical flips because it was lowering performance
        #vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    #v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,h))
    
    return more_images.astype(np.float32)

def get_scaled_imgs(df):
    # Originally found in: https://www.kaggle.com/cbryant/keras-cnn-statoil
    # -iceberg-lb-0-1995-now-0-1516 
    """
    basic function for reshaping and rescaling data as images
    """
    imgs = []
    
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1'])
        band_2 = np.array(row['band_2'])
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)
                                     
def main(unused_argv, use_angles = False):
    # Read in the raw training data
    raw = pd.read_json('data-1/processed/train.json')
    raw.pop('id')
    
    # If using the incidence angles, then drop the samples that don't have
    # the angle. Otherwise, drop the column all together.
    if use_angles:
        inc_angle_na = []
        for i in range(0, raw.shape[0]):
            if raw['inc_angle'].iloc[i] == 'na': inc_angle_na.append(i) 
        raw.drop(labels = inc_angle_na, axis = 0, inplace = True)
    else: raw.pop('inc_angle')
    
    # Scale and add images
    X = get_scaled_imgs(raw)
    y = np.array(raw['is_iceberg'])
    X = get_more_images(X) 
    y = np.concatenate((y, y)) 
 
    # With more time, I would have done the split before adding more images,
    # so that validation occurs only on the original images, not artificials.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                        train_size = 0.8)
    
    # Get the data to predict on
    raw_predict = pd.read_json('data/processed/test.json')
    pred_id = raw_predict.pop('id')
    
    if use_angles:
        inc_angle_na = []
        for i in range(0, raw_predict.shape[0]):
            if raw_predict['inc_angle'].iloc[i] == 'na': inc_angle_na.append(i) 
        raw_predict.drop(labels = inc_angle_na, axis = 0, inplace = True)
    else: raw_predict.pop('inc_angle')
    
    X_predict = get_scaled_imgs(raw_predict).astype(np.float32)
    
    # Build the neural network
    classifier = tf.estimator.Estimator(model_fn = my_model,
                                        model_dir = '/tmp/statoil_nn')
    
    # Log data to the terminal every 1000 steps.
    tensors_to_log = {'probabilities': 'sigmoid_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                                              every_n_iter = 1000)
    
    # Train, evaluate, and predict using the CNN
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
    
    # Prepare the predictions for CSV submission
    y_predict_submission = pd.DataFrame(list(p['probabilities'] 
                                        for p in itertools.islice(
                                         y_predict, X_predict.shape[0]))) 
    submission = pd.concat([pred_id, y_predict_submission], 
                            axis = 1, join = 'inner')
    submission = submission.rename(columns = {1: 'is_iceberg'})
    submission.pop(0)
    submission.to_csv('cnn_submission.csv', index = False)

if __name__ == "__main__":
  tf.app.run()  
