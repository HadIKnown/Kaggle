# Script used for the NOMAD 2018 Transparent Conductors Kaggle competition.
# Details can be found here: https://www.kaggle.com/c/nomad2018-predict-
#                            transparent-conductors
# Top root mean squared log error of 0.0542 with CatBoost, which can be op-
# timized further. Also considering applying a convolutional NN to the geo-
# metry files that aren't used here.

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Import data, store test ID, then normalize output.
train = pd.read_csv('train.csv')
test_submission = pd.read_csv('test.csv')
test_id = test_submission['id']
fe_train = np.log(train['formation_energy_ev_natom'] + 1)
bge_train = np.log(train['bandgap_energy_ev'] + 1)

def findVolume(data):
    # Returns the volume of a parallelepiped
    # https://en.wikipedia.org/wiki/Parallelepiped 
    a = data['lattice_vector_1_ang'] 
    b = data['lattice_vector_2_ang']
    c = data['lattice_vector_3_ang']
    alpha = np.pi * data['lattice_angle_alpha_degree'] / 180.
    beta = np.pi * data['lattice_angle_beta_degree'] / 180.
    gamma = np.pi * data['lattice_angle_gamma_degree'] / 180.
    return (a * b * c *                                                                   
            np.sqrt(1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                      - np.cos(alpha) ** 2
                      - np.cos(beta) ** 2
                      - np.cos(gamma) ** 2))

def usePCA(data, features, label, n_components = 1, random_state = 42):
    # Returns PCA on the features to be reduced.
    # Ultimately, did not help performance. Was used in createFeatures()
    pca = PCA(n_components = n_components, random_state = random_state)
    pca.fit(data[features])
    labels = []
    for i in xrange(0, n_components):
        labels.append(label + '_' + str(i))
    new_features = pd.DataFrame(pca.transform(data[features]),
                                columns = labels)
    return pd.concat([data, new_features], axis = 1, join = 'inner')

def createFeatures(data):
    # Adds the atom_density feature, which is shown to correlate highly
    # with bandgap energy
    volume = pd.Series(findVolume(data))
    data['atom_density'] = data['number_of_total_atoms'] / volume 
    return data
    
def removeFeatures(data, training = True):
    # Drops features. If using PCA, would drop more features. 
    data = data.drop(['id'], axis = 1, inplace = False)
    if training:
        data = data.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'],
                          axis = 1, inplace = False)
    return data

# Apply data transformations.
train = createFeatures(train)
train = removeFeatures(train)
test_submission = createFeatures(test_submission)
test_submission = removeFeatures(test_submission, False)


def regularizeData(data):
    # Returns the regularized data for certain features. Did not improve loss.
    not_scaled = ['spacegroup', 'number_of_total_atoms', 'percent_atom_al',
                  'percent_atom_ga', 'percent_atom_in']
    temp = data[not_scaled]
    data = data.drop(not_scaled, axis = 1, inplace = False)
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    return pd.concat([temp, data], axis = 1, join = 'inner')

def rmsle(p, a):
    # Returns the RMSLE metric that is used for the competition. 
    return np.sqrt(np.mean(np.power(np.log1p(a)-np.log1p(p), 2)))

# CatBoost Implementation
def CBR(X, y):
    # The following are the features to be treated as categorical
    cat_features_labels =['spacegroup', 'number_of_total_atoms']
    cat_features = []
    for feature in cat_features_labels:                                       
        cat_features.append(X.columns.get_loc(feature))
    
    # Split the data for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42) 
    
    # CatBoost documentation uses the Pool structure to feed data 
    # into the algorithm, which (I believe) prevents the use of KFolds
    # from sklearn for cross validation. However, CatBoost is definitely
    # usable sans the Pool structure. Will investigate in the future.
    train_pool = Pool(X_train, y_train, cat_features = cat_features)
    test_pool = Pool(X_test, cat_features = cat_features)
    
    model = CatBoostRegressor(iterations = 2000,
                              depth = 10,
                              learning_rate = 0.001,
                              loss_function = 'RMSE',
                              random_seed = 42,
                              logging_level = 'Silent')
    model.fit(train_pool)
    pred = model.predict(test_pool)
    evaluation = rmsle(pred, y_test)
    all_train_pool = Pool(X, y, cat_features = cat_features)
    model.fit(all_train_pool)
    return evaluation, model

def prepareData(X):
    # Returns the one hot encodings for selected categorical features.
    # For the algorithms besides CatBoost, this must be done.
    onehote = pd.get_dummies(X['spacegroup'])
    X = pd.concat([onehote, X], axis = 1, join = 'inner')
    return X.drop(['spacegroup', 12], axis = 1, inplace = False)

def getBestModel(X, y, model, parameters, cv = 5):
    # Finds the best parameters to use for a given model.
    scorer = make_scorer(rmsle) 
    clf = GridSearchCV(model, parameters, scoring = scorer, cv = cv)
    new_model =  clf.fit(X, y)
    print clf.best_params_
    return new_model

def findRMLSE(X, y, model):
    # Returns the average performance of a model using its best parameters. 
    # Uses KFold from sklearn to generate cross validation sets. 
    rmsles = []
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    for train_i, test_i in kf.split(X):
        X_train, X_test = X.loc[train_i], X.loc[test_i]
        y_train, y_test = y.loc[train_i], y.loc[test_i]
        pred = model.predict(X_test)
        rmsles.append(rmsle(pred, y_test))
    mean_rmsle = np.mean(rmsles)
    return mean_rmsle

# Kernel Ridge Regression Implementation
def KRR(X, y):
    # Found to be used in the literature to perform property prediction.
    X = prepareData(X)
    parameters = {'kernel': ('linear', 'rbf', 'laplacian', 'sigmoid'),                    
                  'alpha': (0.025, 0.035, 0.01, 0.05, 0.1, 0.25, 0.5, 1)}
    model = getBestModel(X, y, KernelRidge(), parameters) 
    mean_rmsle = findRMLSE(X, y, model)
    return mean_rmsle, model

# Random Forest Regression Implementation
def RFR(X, y):
    # Mentioned in the literature, but always good to try as a baseline.
    X = prepareData(X)
    parameters = {'n_estimators': [30, 50, 100, 150],
                  'max_features':[None, 0.3, 0.4, 0.5, 0.8],
                  'max_depth': [None, 5, 8, 10, 12, 15], 
                  'min_samples_split': [2, 3, 4, 5],
                  'min_samples_leaf': [1, 2, 3, 4, 5]}
    model = getBestModel(X, y, RandomForestRegressor(random_state = 42), parameters)
    mean_rmsle = findRMLSE(X, y, model)
    return mean_rmsle, model

# Support Vector Regressor Implementation
def SVM(X, y):
    # Because why not?
    X = prepareData(X)
    parameters = {'C': [1./2**4, 1./2**2, 1./2, 2, 4, 8, 16],
                  'epsilon': [1./2**7, 1./2**6, 1./2**5, 1./2**4, 1./2**2],
                  'kernel': ['linear', 'rbf', 'sigmoid']}
    model = getBestModel(X, y, SVR(), parameters)
    mean_rmsle = findRMLSE(X, y, model)
    return mean_rmsle, model

def makeSubmission(ID, data, model_fe, model_bge, label = 'default'):
    # Output a CSV with the predicted properties.
    if label != 'cat': data = prepareData(data)
    submission = pd.DataFrame()                                            
    submission['id'] = ID
    submission['formation_energy_ev_natom'] = np.expm1(model_fe.predict(data))
    submission['bandgap_energy_ev'] = np.expm1(model_bge.predict(data))
    submission[submission < 0] = 0
    submission.to_csv(label + '_submission.csv', index = False)

# Implement all the models defined above and output their performance.
"""rfr_eval_fe, rfr_fe = RFR(train, fe_train)
rfr_eval_bge, rfr_bge = RFR(train, bge_train)
print 'RFR: ', np.mean([rfr_eval_fe, rfr_eval_bge]), rfr_eval_fe, rfr_eval_bge
makeSubmission(test_id, test_submission, rfr_fe, rfr_bge, 'rfr')

svm_eval_fe, svm_fe = SVM(train, fe_train)
svm_eval_bge, svm_bge = SVM(train, bge_train)
print 'SVM: ', np.mean([svm_eval_fe, svm_eval_bge]), svm_eval_fe, svm_eval_bge
makeSubmission(test_id, test_submission, svm_fe, svm_bge, 'svm')

krr_eval_fe, krr_fe = KRR(train, fe_train)
krr_eval_bge, krr_bge = KRR(train, bge_train)
print 'KRR: ', np.mean([krr_eval_fe, krr_eval_bge]), krr_eval_fe, krr_eval_bge 
makeSubmission(test_id, test_submission, krr_fe, krr_bge, 'krr')                                         
"""
cat_eval_fe, cat_fe = CBR(train, fe_train)
cat_eval_bge, cat_bge = CBR(train, bge_train)
print 'Cat: ', np.mean([cat_eval_fe, cat_eval_bge]), cat_eval_fe, cat_eval_bge 
makeSubmission(test_id, test_submission, cat_fe, cat_bge, 'cat')


