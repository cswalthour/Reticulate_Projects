import os
import my_module
import pandas as pd # You may want to move this import to the top of the file.

def test_invocation():
    features, target = my_module.get_features_and_target(
        csv_file='C:/Users/cswal/Documents/Github/Reticulate_Projects/UC Business Analytics/adult-census.csv',
        target_col='class'
    )

def test_return_types():
    features, target = my_module.get_features_and_target(
        csv_file='C:/Users/cswal/Documents/Github/Reticulate_Projects/UC Business Analytics/adult-census.csv',
        target_col='class'
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
