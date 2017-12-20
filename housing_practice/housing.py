import pandas as pd

raw_data = pd.read_csv('train.csv')

#Narrowed down to 20 features Dec 14

def cleanData(data):
    # Bring back Condition1/2, LowQualFinSF, 
    drop_features = ['MSSubClass', 'LotFrontage', 'Street', 'Alley', 'LotShape', 
                     'LandContour', 'Condition1', 'Condition2', 'BldgType', 
                     'HouseStyle', 'YearBuilt', 'RoofStyle', 'Exterior2nd', 
                     'MasVnrType', 'BsmtExposure', 'Electrical', 'Heating',
                     'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'GarageYrBlt',
                     'GarageFinish', 'GarageCars', 'PavedDrive', 'Fence',
                     'MiscFeature', 'SaleType', 'LotConfig', 'Exterior1st',
                     'MasVnrArea', 'GarageType', 'SaleCondition']
    truncated_data = data.drop(drop_features, axis = 1, inplace = False)
    #Change all the degree'd categoricals into numbers
    seminumberedData = convertLeveledCats(truncated_data)
    #engineerFeatures(truncated_data)

def convertLeveledCats(data):
    from sklearn.preprocessing import LabelEncoder
    #in future, should have separate encoders for the values that
    #are required to adjust in engineerFeatures (BsmtUnfSF, LowQualFinSF, LotArea),  
    #TODO: normalize these encoded values
    le = LabelEncoder()
    data['ExterQual'] = le.fit_transform(data['ExterQual'])
    data['ExterCond'] = le.fit_transform(data['ExterCond'])
    data['BsmtQual'] = le.fit_transform(data['BsmtQual'])
    data['BsmtCond'] = le.fit_transform(data['BsmtCond'])
    data['BsmtFinType1'] = le.fit_transform(data['BsmtFinType1'])
    data['BsmtFinType2'] = le.fit_transform(data['BsmtFinType2'])
    data['GarageQual'] = le.fit_transform(data['GarageQual'])
    data['GarageCond'] = le.fit_transform(data['GarageCond'])
    data['LandSlope'] = le.fit_transform(data['LandSlope'])
    data['KitchenQual'] = le.fit_transform(data['KitchenQual'])
    data['FireplaceQu'] = le.fit_transform(data['FireplaceQu'])
    data['PoolQC'] = le.fit_transform(data['PoolQC'])

def engineerFeatures(data):
    #see if can find better adjust for inner square ft of home
    overall_adjust = data['OverallQual'] * data['OverallCond']
    external_adjust = data['ExterQual'] * data['ExterCond']
    bsmt1_adjust = data['BsmtQual'] * data['BsmtCond'] * data['BsmtFinType1']
    bsmt2_adjust = data['BsmtQual'] * data['BsmtCond'] * data['BsmtFinType2']
    garage_adjust = data['GarageQual'] * data['GarageCond']
    #CONSIDER ADDING LOWQUALFINSF with the lowest adjuster
    alt_sqft = data['LandSlope'] * (overall_adjust * (data['1stFlrSF'] + data['2ndFlrSF']) +
               2 * data['LowQualFinSF'] +
               bsmt1_adjust * data['BsmtFinSF1'] + 
               bsmt2_adjust * data['BsmtFinSF2'] +
               1 * data['BsmtUnfSF'] +
               garage_adjust * data['GarageArea'] + #maybe separate out
                1  * (data['LotArea'] - data['1stFlrSF']))
    alt_bathrooms = (data['BsmtFullBath'] + data['BsmtHalfBath'] + 
                     data['FullBath'] + data['HalfBath'])
    alt_kitchens = data['KitchenQual'] * data['Kitchen']
    alt_fireplaces = data['Fireplaces'] * data['FireplaceQu'] #maybe cut
    outdoors_sqft = (data['WoodDeckSF'] + data['OpenPorchSF'] + 
                     data['EnclosedPorch'] + data['3SsnPorch'] + 
                     data['ScreenPorch'])
    pool_sqft = data['PoolArea'] * data['PoolQC']
    #date_sold = 'YrSold' + 'MoSold' 

cleanData(raw_data)
