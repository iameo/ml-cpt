

# ML-CPT (MACHINE LEARNING-CLEAN PROCESS TRAIN)
---

Get fast inference on your data, with a range of tools/algorithms in your disposal.

Visualization included. Code snippets to see what's running under the hood, too.



[click here: https://mlcpt.herokuapp.com]

***SUPPORTS ONLY CLASSIFICATION TASKS*** (for now..)



### How to Use
---
****use the EXPLORE section****
- The app expects you provide two separate files(train and test) so be sure to include those
- ...the rest is handled automatically, but you can fine-tune as you would like.




### Features
---
- DateTime feature engineering (datetime columns are automatically selected for you)
- Duplicated rows will be dropped
- Columns of type object will be transformed to lower case as to help with cleaning and duplicity(ex: Male and MALE ==> male)
- ID Field selection for final test file (i.e
| ID* | target |
| ------:| -----------:|
| customerID001   | 1 |
| customerID00109 | 0 |
| .    | . |
| .    | . |

- Target column is automatically collected (looks for "target", "claim", "prediction", "response" as shown above; you can also pick the desired target column)
- Retain Missing Data as you want(default is 50%) and you can also choose how to treat the missing data(default is mode)
- Visualization
- Selecting features to drop
- Monotonic or/and unique data dropping (ID has been stored before this)
- GetDummies on categorical features (drop_first is True)
- Choose scaler (Standardization or Normalization)
- Download your processed dataset(train, test)
- Auto dataset split (60% - Train, 30% - Validation and 10% - Test)
- (NEW*) SMOTE, RANDOM OVER/UNDER SAMPLER FOR IMBALANCED DATASET!
- Algorithm selection (Catboost, Knn, RandomForest and Xgboost)
- Detailed report on prediction
- Save Test prediction to obtain a baseline model score on your Hackathon


### CONTRIBUTION
---
If your dataset fails to parse correctly then it's a sign for you to contribute to the project. Be sure to checkout on a new branch for any feature/fix you add. :)

To contribute: [here](https://github.com/iameo/ml-cpt)

### PERFORMANCE
---
Did you get a score with our test prediction? Kindly include that below(screenshot or in writing)

- UmojaHack Nigeria: AXA Vehicle Insurance Claim Challenge by UmojaHack Africa (on ZINDI) - ~37.6% (72nd ranking) [base selections]


#### UNSUPPORTED DATASET 
If you have a dataset(classification based) that fails using this app, kindly include it [here](https://github.com/iameo/ml-cpt) as a PR.
