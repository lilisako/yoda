import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from yoda import FeatureEngineering



df = pd.read_csv('../data/titanic/train.csv')
train, test = train_test_split(df, test_size=0.33, random_state=42)
train, test = FeatureEngineering.count_encoding(train, test, ['Pclass'])