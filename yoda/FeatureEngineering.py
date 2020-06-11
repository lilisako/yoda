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