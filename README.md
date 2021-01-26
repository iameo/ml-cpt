

# ML-CPT

I made this web app to assist ML begineers, as well as experts to run a model fast with adequate visualization to aid.


SUPPORTS ONLY CLASSIFICATION TASKS (for now..)




### How to Use
- The app expects that you have two separate files(train and test) so be sure to include those
- ...the rest is handled as long as the required model isn't tasking(very little wrangling onboard at the moment)




### Features
- DateTime feature engineering ("auto" selects but you get to pick)
- Duplicated columns and rows will be dropped
- Columns of type object will be transformed to lower case as to help with cleaning and duplicity(ex: Male and MALE ==> male)
- ID Field selection for final test file (i.e ID | prediction)
- Target column "auto" collected (looks for "target", "claim", "prediction", "response"; you can also pick the desired target column)
- Retain Missing Data as you want(default is 50%) and you can also choose how to treat the missing data(default is mode)
- Visualization
- Selecting features to drop
- Monotonic or/and unique data dropping (ID has been stored before this)
- GetDummies on categorical features (drop_first is True)
- Choose scaler (Standardization or Normalization)
- Download your processed dataset
- Auto dataset split (60% - Train, 30% - Validation and 10% - Test)
- Algorithm selection (Catboost, Knn, RandomForest and Xgboost)
- Detailed report on prediction