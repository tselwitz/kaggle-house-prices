import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_df(df, y_col):
    y = torch.tensor(df[y_col]).type(torch.float32)
    df.pop(y_col)
    features_to_encode = df.keys()
    df["MSSubClass"] = df["MSSubClass"].astype("object")
    mm_scaler = MinMaxScaler()
    for feature in features_to_encode:
        if df[feature].dtype == "object":
            df = encode_and_bind(df, feature)
        elif df[feature].dtype in ["float64", "int64"]:
            temp = np.array(df[feature]).reshape(1, -1)
            df[feature] = mm_scaler.fit_transform(temp).T
    X = df.to_numpy(np.float64)
    X = torch.tensor(X).type(torch.float32)
    return X, y

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

def clean(df):
    for col in df:
        if df[col].dtype == "object":
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].interpolate()
    return df