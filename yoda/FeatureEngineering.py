import pandas as pd 
import numpy as np 

def count_encoding(train: pd.DataFrame, test: pd.DataFrame, target: dict):
    df = pd.concat([train,test],sort=False).reset_index(drop=True)
    l = len(train)
    for col in target:
        count_map = df[col].value_counts().to_dict()
        df['count_enc_'+col] = df[col].map(count_map)
    train = df[l:].reset_index(drop=True)
    test = df[:l]
    return train,test

def label_encoding(train: pd.DataFrame, test: pd.DataFrame, target: dict):
    df = pd.concat([train,test],sort=False).reset_index(drop=True)
    l = len(train)
    for col in target:
        df[col] = df[col].astype('category')
        df['label_enc_'+col] = df[col].cat.codes
    train = df[l:].reset_index(drop=True)
    test = df[:l]
    return train,test

def fillna(train: pd.DataFrame, test:pd.DataFrame, target: dict, mode='mean'):
    df = pd.concat([train,test],sort=False).reset_index(drop=True)
    l = len(train)
    if mode == 'mean':
        df[target] = df[target].fillna(df.mean())
    elif mode == 'median':
        df[target] = df[target].fillna(df.median())
    elif mode == 'mode':
        df[target] = df[target].fillna(df.mode())
    else:
        return None
    train = df[l:].reset_index(drop=True)
    test = df[:l]
    return train,test

def info(df: pd.DataFrame()):
    print("===== NULL COUNT =====")
    print(df.isna().sum())
    print("===== INFO =====")
    print(df.info())
    print("===== DESCRIBE ======")
    print(df.describe())

def categorical_info(df: pd.DataFrame, target: str, categorical: str):
    print(df[[target, categorical]].groupby([categorical], as_index=False).mean())