import pandas as pd

raw_data = pd.read_csv('train.csv') 

#Narrowed down to 20 features Dec 14

def cleanData(data):
    # Bring back Condition1/2, LowQualFinSF, 
    # Bring back Id for........ maybe submission
    drop_features = ['MSSubClass', 'LotFrontage', 'Street', 'Alley', 'LotShape', 
                     'LandContour', 'Condition1', 'Condition2', 'BldgType', 
                     'HouseStyle', 'YearBuilt', 'RoofStyle', 'Exterior2nd', 
                     'MasVnrType', 'BsmtExposure', 'Electrical', 'Heating',
                     'GrLivArea', 'BedroomAbvGr', 'GarageYrBlt',
                     'GarageFinish', 'GarageCars', 'PavedDrive', 'Fence',
                     'MiscFeature', 'SaleType', 'LotConfig', 'Exterior1st',
                     'MasVnrArea', 'GarageType', 'SaleCondition', 'Id',
                     'TotalBsmtSF']
    truncated_data = data.drop(drop_features, axis = 1, inplace = False)
    #Change all the degree'd categoricals into numbers
    seminumberedData = convertLeveledCats(truncated_data)
    engineerFeatures(truncated_data)

def convertLeveledCats(data):
    from sklearn.preprocessing import LabelEncoder
    #in future, should have separate encoders for the values that
    #are required to adjust in engineerFeatures (BsmtUnfSF, LowQualFinSF, LotArea),  
    #TODO: normalize these encoded values
    """le = LabelEncoder()
    le.fit(['Po', 'Fa', 'TA', 'Gd', 'Ex'])
    print le.classes_
    data['ExterQual'] = le.transform(data['ExterQual'])
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
    data = data.replace({'ExterQual': qualcond5, 
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
                         'LandSlope': qualcondland})
    fillNA = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual', 
              'FireplaceQu', 'GarageQual', 'GarageCond', 'HeatingQC', 'PoolQC', 
              'BsmtFinType1', 'BsmtFinType2', 'LandSlope']
    data[fillNA] = data[fillNA].fillna(value = 0)
    temp = data[fillNA]
    print temp

def engineerFeatures(data):
    #see if can find better adjust for inner square ft of home
    overall_adjust = data['OverallQual'] * data['OverallCond']
    external_adjust = data['ExterQual'] * data['ExterCond']
    bsmt1_adjust = data['BsmtQual'] * data['BsmtCond'] * data['BsmtFinType1']
    bsmt2_adjust = data['BsmtQual'] * data['BsmtCond'] * data['BsmtFinType2']
    garage_adjust = data['GarageQual'] * data['GarageCond']
    #CONSIDER ADDING LOWQUALFINSF with the lowest adjuster
    #instead of creating variable and then adding new column, add immediately
    data['alt_sqft'] = (data['LandSlope'] * 
                       (overall_adjust * (data['1stFlrSF'] + data['2ndFlrSF']) +
                        2 * data['LowQualFinSF'] +
                        bsmt1_adjust * data['BsmtFinSF1'] + 
                        bsmt2_adjust * data['BsmtFinSF2'] +
                        1 * data['BsmtUnfSF'] +
                        garage_adjust * data['GarageArea'] + #maybe separate out
                        1  * (data['LotArea'] - data['1stFlrSF'])))
    data['alt_bathrooms'] = (data['BsmtFullBath'] + data['BsmtHalfBath'] + 
                             data['FullBath'] + data['HalfBath'])
    data['alt_kitchens'] = data['KitchenQual'] * data['KitchenAbvGr']
    data['alt_fireplaces'] = data['Fireplaces'] * data['FireplaceQu'] #maybe cut
    data['outdoors_sqft'] = (data['WoodDeckSF'] + data['OpenPorchSF'] + 
                             data['EnclosedPorch'] + data['3SsnPorch'] + 
                             data['ScreenPorch'])
    data['pool_sqft'] = data['PoolArea'] * data['PoolQC']
    data['date_sold'] = 365.25 * data['YrSold'] + 30.44 * data['MoSold']
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
    print truncated_data
cleanData(raw_data)
