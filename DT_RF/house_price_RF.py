import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from RandomForestRegressor import RandomForestRegressor

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

le = [
    "MSZoning",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "BldgType",
    "HouseStyle",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
]

ohe = [
    "MSSubClass",
    "Street",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    'Alley'
]

na_is_none_cols = [
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", 'Alley'
]

n_train = df_train.shape[0]
y_train = df_train['SalePrice'].values
df_train_features = df_train.drop(columns=['SalePrice'])

df_all = pd.concat([df_train_features, df_test], axis=0).reset_index(drop=True)

na_is_none_cols = [c for c in na_is_none_cols if c in df_all.columns]
df_all[na_is_none_cols] = df_all[na_is_none_cols].fillna("None")

for col in df_all.columns:
    if is_numeric_dtype(df_all[col]):
        df_all.loc[:, col] = df_all[col].fillna(df_all[col].median())
    else:
        df_all.loc[:, col] = df_all[col].fillna(df_all[col].mode()[0])

for col in le:
    encoder = LabelEncoder()
    df_all[col] = encoder.fit_transform(df_all[col].astype(str))

if 'MSSubClass' in ohe:
    df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)

df_all = pd.get_dummies(df_all, columns=ohe, drop_first=False)

X_train = df_all.iloc[:n_train].values
X_test = df_all.iloc[n_train:].values

print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước y_train: {y_train.shape}")
print(f"Kích thước X_test:  {X_test.shape}")

forest = RandomForestRegressor(n_estimators=100, max_depth=15, max_features='third')
forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
print(y_predict)

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': y_predict
})

submission.to_csv('submission.csv', index=False)
