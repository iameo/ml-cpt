

# ML-CPT (MACHINE LEARNING-CLEAN PROCESS TRAIN)
---

I made this web app to assist ML Beginners, as well as Experts to run a model fast with adequate visualization to aid.


SUPPORTS ONLY CLASSIFICATION TASKS (for now..)




### How to Use
---
- The app expects that you have two separate files(train and test) so be sure to include those
- ...the rest is handled as long as the required model isn't tasking(very little wrangling onboard at the moment)




### Features
---
- DateTime feature engineering ("auto" selects but you get to pick)
- Duplicated columns and rows will be dropped
- Columns of type object will be transformed to lower case as to help with cleaning and duplicity(ex: Male and MALE ==> male)
- ID Field selection for final test file (i.e
| ID* | target |
| ------:| -----------:|
| customerID001   | 1 |
| customerID00109 | 0 |
| .    | . |

- Target column "auto" collected (looks for "target", "claim", "prediction", "response" as shown above; you can also pick the desired target column)
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


### Tip
---
If your dataset fails to parse correctly then it's a sign for you to contribute to the project. Be sure to checkout on a new branch for any feature/fix you add. :)