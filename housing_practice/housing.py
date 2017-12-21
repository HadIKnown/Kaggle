import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

raw_data = pd.read_csv('train.csv') 

#to adjust final sale price prediction
misc_values = raw_data['MiscVal']

#Narrowed down to 20 features Dec 14

def regularize(column):
    return (column - np.mean(column)) / (max(column) - min(column))

def cleanData(data):
    # Bring back Condition1/2, LowQualFinSF, 
    # Bring back Id for........ maybe submission
    # Bring back Utilities?
    drop_features = ['MSSubClass', 'LotFrontage', 'Street', 'Alley', 'LotShape', 
                     'LandContour', 'Condition1', 'Condition2', 'BldgType', 
                     'HouseStyle', 'YearBuilt', 'RoofStyle', 'Exterior2nd', 
                     'MasVnrType', 'BsmtExposure', 'Electrical', 'Heating',
                     'GrLivArea', 'BedroomAbvGr', 'GarageYrBlt',
                     'GarageFinish', 'GarageCars', 'PavedDrive', 'Fence',
                     'MiscFeature', 'SaleType', 'LotConfig', 'Exterior1st',
                     'MasVnrArea', 'GarageType', 'SaleCondition', 'Id',
                     'TotalBsmtSF', 'MiscVal', 'Utilities']
    truncated_data = data.drop(drop_features, axis = 1, inplace = False)
    #Change all the degree'd categoricals into numbers
    seminumberedData = convertLeveledCats(truncated_data)
    cleanedData = engineerFeatures(truncated_data)
    return cleanedData

def convertLeveledCats(data):
    """
    from sklearn.preprocessing import LabelEncoder
    #in future, should have separate encoders for the values that
    #are required to adjust in engineerFeatures (BsmtUnfSF, LowQualFinSF, LotArea),  
    #TODO: normalize these encoded values
    le = LabelEncoder()
    data['ExterQual'] = le.fit_transform(data['ExterQual'])
    #print data['ExterQual']
    data['ExterCond'] = le.fit_transform(data['ExterCond'])
    data['BsmtQual'] = le.fit_transform(data['BsmtQual'])
    data['BsmtCond'] = le.fit_transform(data['BsmtCond'])
    data['BsmtFinType1'] = le.fit_transform(data['BsmtFinType1'])
    data['BsmtFinType2'] = le.fit_transform(data['BsmtFinType2'])
    l3 = LabelEncoder()
    data['GarageQual'] = le.fit_transform(data['GarageQual'])
    data['GarageCond'] = le.fit_transform(data['GarageCond'])
    data['LandSlope'] = le.fit_transform(data['LandSlope'])
    data['KitchenQual'] = le.fit_transform(data['KitchenQual'])
    data['FireplaceQu'] = le.fit_transform(data['FireplaceQu'])
    data['PoolQC'] = le.fit_transform(data['PoolQC'])
    """
    #evidently, LabelEncoder is trash so:
    qualcond5 = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    qualcondPool = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}                   
    qualcondbsmtfin = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 
                       'GLQ': 6}
    qualcondland = {'Sev': 1, 'Mod': 2, 'Gtl': 3}
    data.replace({'ExterQual': qualcond5, 
                  'ExterCond': qualcond5,
                  'BsmtQual': qualcond5,
                  'BsmtCond': qualcond5,
                  'KitchenQual': qualcond5,
                  'FireplaceQu': qualcond5,
                  'GarageQual': qualcond5,
                  'GarageCond': qualcond5,
                  'HeatingQC': qualcond5,
                  'PoolQC': qualcondPool,
                  'BsmtFinType1': qualcondbsmtfin,
                  'BsmtFinType2': qualcondbsmtfin,
                  'LandSlope': qualcondland, 
                  'CentralAir': {'Y': 1, 'N': 0}}, inplace = True)
    fillNA = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual', 
              'FireplaceQu', 'GarageQual', 'GarageCond', 'HeatingQC', 'PoolQC', 
              'BsmtFinType1', 'BsmtFinType2', 'LandSlope', 'CentralAir',
               #following line added as bandaid for odd NaN entries 
              'MSZoning', 'Neighborhood', 'RoofMatl', 'Foundation', 'Functional']
    #eliminate NAs leftover
    data[fillNA] = data[fillNA].fillna(value = 0)
    
    #normalize scalers
    """for col in fillNA:
        if data[col].isnull().values.any():
            data[col] = data[col].fillna(value = 0)
"""

def engineerFeatures(data):
    #see if can find better adjust for inner square ft of home
    #NORMALIZE ALL THESE adjusts
    overall_adjust = regularize(data['OverallQual'] * data['OverallCond'])
    external_adjust = regularize(data['ExterQual'] * data['ExterCond'])
    bsmt1_adjust = regularize(data['BsmtQual'] * data['BsmtCond'] * 
                              data['BsmtFinType1'])
    bsmt2_adjust = regularize(data['BsmtQual'] * data['BsmtCond'] * 
                              data['BsmtFinType2'])
    garage_adjust = regularize(data['GarageQual'] * data['GarageCond'])
    data['PoolQC'] = regularize(data['PoolQC'])                                      

    #CONSIDER ADDING LOWQUALFINSF with the lowest adjuster
    #instead of creating variable and then adding new column, add immediately
    ####there might be better way than hardcoding the weights
    data['alt_sqft'] = (data['LandSlope'] * 
                       (overall_adjust * (data['1stFlrSF'] + data['2ndFlrSF']) +
                        0.2 * data['LowQualFinSF'] +
                        bsmt1_adjust * data['BsmtFinSF1'] + 
                        bsmt2_adjust * data['BsmtFinSF2'] +
                        0.1 * data['BsmtUnfSF'] +
                        garage_adjust * data['GarageArea'] + #maybe separate out
                        0.1 * (data['LotArea'] - data['1stFlrSF'])))
    data['alt_bathrooms'] = (data['BsmtFullBath'] + data['BsmtHalfBath'] + 
                             data['FullBath'] + data['HalfBath'])
    data['alt_kitchens'] = data['KitchenQual'] * data['KitchenAbvGr']
    data['alt_fireplaces'] = data['Fireplaces'] * data['FireplaceQu'] #maybe cut
    data['outdoors_sqft'] = (data['WoodDeckSF'] + data['OpenPorchSF'] + 
                             data['EnclosedPorch'] + data['3SsnPorch'] + 
                             data['ScreenPorch'] + 
                             (data['PoolArea'] * data['PoolQC']))
    #data['pool_sqft'] = data['PoolArea'] * data['PoolQC'] add back later?
    data['date_sold'] = 365.25 * data['YrSold'] + 30.44 * data['MoSold']
    data['date_sold'] = data['date_sold'] - min(data['date_sold'])
    #drop old ones
    used_features = ['LandSlope', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                     'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageArea',
                     'LotArea', '1stFlrSF', 'BsmtFullBath', 'BsmtHalfBath',
                     'FullBath', 'HalfBath', 'KitchenQual', 'KitchenAbvGr',
                     'Fireplaces', 'FireplaceQu', 'WoodDeckSF', 'OpenPorchSF',
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                     'PoolQC', 'YrSold', 'MoSold', 'OverallCond',
                     'OverallQual', 'ExterQual', 'ExterCond', 'BsmtQual',
                     'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'GarageQual',
                     'GarageCond']
    truncated_data = data.drop(used_features, axis = 1, inplace = False)

    return truncated_data

clean_data = cleanData(raw_data)
y = clean_data['SalePrice']
clean_data.drop('SalePrice', axis = 1, inplace = True)
X = clean_data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

###### CatBoost ######
cat_features_labels = ['Functional', 'Foundation', 'RoofMatl', 
                       'Neighborhood', 'MSZoning'] 
cat_features = []
for feature in cat_features_labels:
    cat_features.append(X_train.columns.get_loc(feature))
train_pool = Pool(X_train, y_train, cat_features = cat_features)
test_pool = Pool(X_test, cat_features = cat_features)
model = CatBoostRegressor(iterations = 500, depth = 5, learning_rate = 0.1,
                          loss_function = 'RMSE')
model.fit(train_pool)
pred = model.predict(test_pool)
print mean_squared_error(y_test, pred)

##### Submission #######

test_data = pd.read_csv('test.csv')
Id = pd.DataFrame(test_data['Id'])
clean_data = cleanData(test_data)
submission_test_pool = Pool(clean_data, cat_features = cat_features)
pred_final = pd.DataFrame(model.predict(submission_test_pool))
final_output = pd.concat([Id, pred_final], axis = 1, join = 'inner')
final_output.to_csv('submissionCat.csv')
