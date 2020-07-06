import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def rf_classification(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame):
    random_forest = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
    random_forest.fit(train_x, train_y)
    y_pred = random_forest.predict(test_x)
    accuracy_random_forest = accuracy_score(test_y, y_pred)
    print('Accuracy: {}'.format(accuracy_random_forest))
    print(classification_report(test_y, y_pred))


