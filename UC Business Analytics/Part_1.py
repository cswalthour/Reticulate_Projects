
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######### READ-IN DATA #########
raw_data = pd.read_csv('C:/Users/cswal/Documents/Github/Reticulate_Projects/UC Business Analytics/ames.csv')

# Inspect prelim structure of dataset
print(raw_data.info())

# Isolate target/response variable
target_column = raw_data["Sale_Price"]

# Isolate predictive features
features = raw_data.drop(columns = target_column)

# Determine numeric features
numeric_columns = features.select_dtypes(include=np.number).columns.values
# Determine categorical features
categorical_columns = features.drop(columns=numeric_columns).columns.values

# Print statement analyzing numerical/categorical features strings
print(f'''
There are {features.shape[0]} observations and {features.shape[1]} features.

Numeric features: {', '.join(numeric_columns)}.

Categorical features: {', '.join(categorical_columns)}.
''')

######### EDA #########
import math
import seaborn as sns

# Review target feature distribution
target_column.hist(figsize=(10, 5), bins=30);
plt.show()

# For loop intended to review all numeric features at once via scatterplot
i,j = 0, 0
plots_per_row = 5
fig, axs = plt.subplots(math.ceil(len(numeric_columns)/plots_per_row),
  plots_per_row, figsize = (20, 60))

for col in numeric_columns:
  axs[i][j].scatter(raw_data[target_column], 
                        raw_data[col], s = 3)
  axs[i][j].set_ylabel(col)
  j+=1
  if j%plots_per_row==0:
    i+=1
    j=0
plt.show()

# For loop intended to review all numeric features, one-at-a-time via histogram
for i in numeric_columns:
    plt.figure(figsize=(12,5))
    plt.title(i)
    sns.histplot(data=raw_data[i])
    plt.show()
    
# For loop intended to review all categorical features, one-at-a-time via boxplot
for i in categorical_columns:
    plt.figure(figsize=(12,5))
    plt.title(i)
    raw_data.boxplot(column='Sale_Price', by=i, figsize=(10,5), rot=45);
    plt.show()


######### TRAIN AND TESTING OF REGRESSION MODELS #########

import my_module
from sklearn.model_selection import train_test_split

# Create new features and new target
new_features, new_target = my_module.get_features_and_target(
  'C:/Users/cswal/Documents/Github/Reticulate_Projects/UC Business Analytics/ames.csv',
  'Sale_Price')
  
# Assign data structures specific to training and testing
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
  new_features,
  new_target, 
  random_state=123,
  test_size=.25)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

def run_exps(X_train_new: pd.DataFrame, 
  y_train_new: pd.DataFrame, 
  X_test_new: pd.DataFrame, 
  y_test_new: pd.DataFrame) -> pd.DataFrame:
    '''Lightweight script to test many models and find winners:
      param X_train: 
        training split:
          param y_train: training target vector:
            param X_test: 
              test split:
                param y_test: 
                  test target vector:
                    return: DataFrame of predictions
    '''
    dfs = []
    models = [
          ('LinReg', LinearRegression()), 
          ('RF', RandomForestRegressor()),
          ('KNN', KNeighborsRegressor())
          ]
        
    results = []
    
    names = []
        
    #scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    #target_names = ['malignant', 'benign']
        
    for name, model in models:
      
      kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
      cv_results = model_selection.cross_validate(model, X_train_new, y_train_new, cv=kfold)
      clf = model.fit(X_train_new, y_train_new)
      #y_pred = clf.predict(X_test)
      print(name)
      
      #Score our model on test data
      accuracy = model.score(X_test_new, y_test_new)
      
      print(f'The test R-Sqaure using {name} is {round(accuracy, 4) * 100}%')
      
      results.append(cv_results)
      
      names.append(name)
      
      this_df = pd.DataFrame(cv_results)
      
      this_df['model'] = name
      
      dfs.append(this_df)
      
      final = pd.concat(dfs, ignore_index=True)
      
      return final
    
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

lm_model_scaled = make_pipeline(StandardScaler(), LinearRegression())
lm_model_scaled.fit(X_train_new, y_train_new)
lm_model_scaled.score(X_test_new, y_test_new)

######## PREP FOR FEATURE ENGINEERING ########

# get columns of interest
ordinal_columns = features.filter(regex='Qual').columns

# Determine categorical features
features_cleansed = features.drop(columns=ordinal_columns)

# Determine numeric features
numeric_columns = features_cleansed.select_dtypes(include=np.number).columns.values

# Determine categorical features
categorical_columns = features_cleansed.drop(columns=numeric_columns).columns.values

# replace unique values in our ordinal columns (i.e. "No_basement", "No_garage") with 'NA'
for col in ordinal_columns:
  print(f'Evaluating column: {col}')
  features[col] = features[col].replace(to_replace='No_.*', value='NA', regex=True)
    
# split full feature set (numeric, categorical, & ordinal features) into train & test sets
X_train_cleansed, X_test_cleansed, y_train_cleanse, y_test_cleansed = train_test_split(
  features, target_column, random_state=123)

