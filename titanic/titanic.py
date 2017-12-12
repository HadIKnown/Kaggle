###############################################################################
### This was written for my first  Kaggle competition: Titanic. Stopped at score
### of 0.79904 accuracy. Reports indicate the top, "non-cheating" scores hover
### around 0.85. I believe crafty feature engineering, particularly through
### combining Age and Fare, as well as incorporating titles provided in the
### Names feature, would bump up my score a bit. - Yeshar Hadi, 12/12/2017
##############################################################################

### Import relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

### Read in the training data

data = pd.read_csv('train.csv')

### Helper Functions

def cleanData(data):
    ### Data cleaning activities executed here. Could be broken up further.
    
    # Drop features that won't be used. Could use Name in future.
    data.drop(['Name', 'PassengerId', 'Ticket'], axis = 1, inplace = True)

    # Change 'female' to 1 and 'male' to 0
    data['Sex'] = 1 * (data['Sex'] == 'female')

    # Make cabin only the level, not room number. Could add back in future.
    cabin = data['Cabin']
    length = len(cabin)
    newCabin = []
    for i in range(0, length):
        datum = cabin[i]
        # Assume missing cabin means passenger didn't have one.
        if (not isinstance(datum, basestring) and np.isnan(datum)): 
            newCabin.append('0')
        else: 
            newCabin.append(datum[0])
    data['Cabin'] = newCabin

    # Set NaN for given features to 0. 
    data[['Fare', 'Embarked']] = data[['Fare', 'Embarked']].fillna(value = 0)

    # Turn categoricals (Embarked, Cabin) into booleans
    data = pd.get_dummies(data)
    
    # Eliminate essentially equal or redundant features created from get_dumm
    eliminate = ['Cabin_0', 'Cabin_T', 'Embarked_0', 'Embarked_Q', 'Cabin_G', 'Cabin_F']
    parameters = []
    for feature in eliminate:
        if feature in data.axes[1]: parameters.append(feature)
    data.drop(parameters, axis = 1, inplace = True)
    
    # Fix NaN in Age using linear regression, if NaN exists
    if data.isnull().values.any():
        data = regressAge(data)
    
    # Normalize Age, Fare
    features = ['Age','Fare']
    for feature in features:
        data[feature] = np.log(data[feature] + 1) # +1 in case there are 0's
    
    return data

def regressAge(data):
    ### Approximate the missing values for Age

    # Separate data
    y = data['Age']
    data.drop(['Age'], axis = 1, inplace = True)
    X = data
    
    # Create the regression object
    lr = LinearRegression()
    NA = pd.isnull(y)
    notNA = ~NA
    lr.fit(X[notNA], y[notNA])

    # Fidget with dataframes to rebuild data into expected form                               
    y = y.fillna(value = dict(zip(NA.index[NA == True], abs(lr.predict(X[NA])))))
    return pd.concat([X, y], axis = 1, join = 'inner')

def doPCA(data):
    ### Used in data exploration to understand most important features
    from sklearn.decomposition import PCA                                                    
    pca = PCA(n_components = 4)
    data = pca.fit_transform(data) 
    print pca.explained_variance_ratio_
    print data
    return data

### Pre-processing of data                                                                          

data = cleanData(data)

### Visualizations to explore data a bit more
"""
from pandas.tools.plotting import scatter_matrix
#scatter_matrix(data, alpha = 0.3, figsize = (14, 8), diagonal = 'kde')
from seaborn import heatmap
#heatmap(data.corr(), annot = True)
#plt.show()
"""

### Separate data for classification
y = data['Survived']
data.drop(['Survived'], axis = 1, inplace = True)
X = data                                                                                            
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

### Random Forest Classifier

# Set up grid to optimize parameters for RFC
rfc = RandomForestClassifier()
scorer = make_scorer(accuracy_score)
parameters = {'n_estimators': [10], 'criterion': ['entropy', 'gini'], 
              'min_samples_split': [10, 15, 20], 'min_samples_leaf': [1, 2, 3]}
grid_clf = GridSearchCV(rfc, parameters, scoring = scorer, cv = 10)
best_clf = grid_clf.fit(X_train, y_train).best_estimator_
pred = best_clf.predict(X_test)
print 'RFC:', accuracy_score(y_test, pred)

### Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
print 'NB:', accuracy_score(y_test, pred)

### Support Vector Machine
svm = LinearSVC()
scorer = make_scorer(accuracy_score)
parameters = {}
grid_svm = GridSearchCV(svm, parameters, scoring = scorer, cv = 10)
best_svm = grid_svm.fit(X_train, y_train).best_estimator_
pred = best_svm.predict(X_test)
print 'SVM:', accuracy_score(y_test, pred)

### Predict final submission
# Outputs the SVM and RF predictions separately.
data_final = pd.read_csv('test.csv')
PassID =  pd.DataFrame(data_final['PassengerId'])
data_final = cleanData(data_final)
pred_final = pd.DataFrame(best_clf.predict(data_final))
pred_final_svm = pd.DataFrame(best_svm.predict(data_final))
output_final = pd.concat([PassID, pred_final], axis = 1, join = 'inner')
output_final_svm = pd.concat([PassID, pred_final_svm], axis = 1, join = 'inner')
output_final.to_csv('submissionRFC.csv')
output_final_svm.to_csv('submissionSVM.csv')
