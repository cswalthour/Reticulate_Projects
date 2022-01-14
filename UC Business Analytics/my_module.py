import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_features_and_target(csv_file, target_col):
    '''Split a CSV into a DF of numeric features and a target column.'''
    
    raw_data = pd.read_csv(csv_file)
    
    raw_features = raw_data.drop(columns=target_col)
    numeric_features = raw_features.select_dtypes(np.number)
    feature_cols = numeric_features.columns.values

    features = raw_data[feature_cols]
    target = raw_data[target_col]
    
    return (features, target)
