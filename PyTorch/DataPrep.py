import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_df(df, y_col):
    y = torch.tensor(df[y_col]).type(torch.float32)
    df.pop(y_col)
    features_to_encode = df.keys()
    for feature in features_to_encode:
        if df[feature].dtype == "object":
            df = encode_and_bind(df, feature)
    X = df.to_numpy(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X).type(torch.float32)
    return X, y

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

def clean(df):
    tolerance = int(len(df) * 0)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # print(col, df[col].isnull().sum())
            if df[col].isnull().sum() > tolerance:
                df.pop(col)
            else:
                df[col].replace(np.nan, 0)
    return df
        