
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
numeric_columns = features.select_dtypes(include=np.number)
# Determine categorical features
categorical_columns = features.drop(columns=numeric_columns)

# Print statement analyzing numerical/categorical features strings
#print(f'''
#There are {features.shape[0]} observations and {features.shape[1]} features.

#Numeric features: {', '.join(numeric_columns)}.

#Categorical features: {', '.join(categorical_columns)}.
#''')

######### EDA #########
import math
import seaborn as sns

# Review target feature distribution
target_column.hist(figsize=(10, 5), bins=30);
plt.show()

# For loop intended to review all numeric features at once via scatterplot
i,j = 0, 0
plots_per_row = 5
fig, axs = plt.subplots(math.ceil(len(numeric_columns.columns)/plots_per_row),
  plots_per_row, figsize = (20, 60))

for col in numeric_columns.columns:
  axs[i][j].scatter(raw_data[target_column], 
                        raw_data[col], s = 3)
  axs[i][j].set_ylabel(col)
  j+=1
  if j%plots_per_row==0:
    i+=1
    j=0
plt.show()

# For loop intended to review all numeric features, one-at-a-time via histogram
for i in numeric_columns.columns:
    plt.figure(figsize=(12,5))
    plt.title(i)
    sns.histplot(data=raw_data[i])
    plt.show()
    
# For loop intended to review all categorical features, one-at-a-time via boxplot
for i in categorical_column.columnss:
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
numerical_columns = numeric_columns.columns
ordinal_columns = categorical_columns.filter(regex='Qual').columns
categorical_columns = categorical_columns.drop(columns=ordinal_columns).columns

# replace unique values in our ordinal columns (i.e. "No_basement", "No_garage") with 'NA'
for col in ordinal_columns:
  features[col] = features[col].replace(to_replace='No_.*', value='NA', regex=True)
    
# split full feature set (numeric, categorical, & ordinal features) into train & test sets
X_train, X_test, y_train, y_test = train_test_split(features, target_column, random_state=123)

######## FEATURE ENGINEERING ########

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# create our numeric, categorical, and ordinal preprocessor encoders
numerical_preprocessor = StandardScaler()
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

# Define category levels that align with categorical variables that
# include the term "qual" in feature title
ordinal_categories = [
    "NA", "Very_Poor", "Poor", "Fair", "Below_Average", "Average", "Typical",
    "Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"
]

# Impute category levels into nested list for all ordinal columns 
list_of_ord_cats = [ordinal_categories for col in ordinal_columns]

# Deploy encoder only to ordinal columns
ordinal_preprocessor = OrdinalEncoder(categories=list_of_ord_cats)

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('standard_scaler', numerical_preprocessor, numerical_columns),
    ('one_hot_encoder', categorical_preprocessor, categorical_columns),
    ('ordinal_encoder', ordinal_preprocessor, ordinal_columns),
])

######## MODEL FITTING ########

lm_model = make_pipeline(preprocessor, LinearRegression())

_ = lm_model.fit(X_train, y_train)

lm_model.score(X_test_cleansed, y_test_cleansed)

######## MODEL EVALUATION & SELECTION ########

# packages used
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# Pipeline object to chain together modeling processes
rf_model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100))

# fit our model
_ = rf_model.fit(X_train, y_train)

# applying mean squared error in a random forest regressor k-fold cross validation procedure
cross_validate(rf_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

from scipy.stats import loguniform

class loguniform_int:
  """Integer valued version of the log-uniform distribution"""
  def __init__(self, a, b):
    self._distribution = loguniform(a, b)
    
  def rvs(self, *args, **kwargs):
    """Random variable sample"""
    return self._distribution.rvs(*args, **kwargs).astype(int)

from sklearn.pipeline import Pipeline

# create preprocessor & modeling pipeline
rf = RandomForestRegressor(random_state=123)
pipeline = Pipeline([('prep', preprocessor), ('rf', rf)])

# specify hyperparameter distributions to randomly sample from
param_distributions = {
    'rf__n_estimators': loguniform_int(50, 1000),
    'rf__max_features': loguniform(.1, .8),
    'rf__max_depth': loguniform_int(2, 30),
    'rf__min_samples_leaf': loguniform_int(1, 100),
    'rf__max_samples': loguniform(.5, 1),
}

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_distributions, 
    n_iter=25,
    cv=5, 
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1,
)

results = random_search.fit(X_train, y_train)

results.best_score_

results.best_params_



