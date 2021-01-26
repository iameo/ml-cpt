#streamlit
import streamlit as st

#computational and dataframe
import pandas as pd
import numpy as np
import numbers

#plots
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

#sklearn - data, metrics and algos
from sklearn import datasets #tests
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics


import sklearn
import sys

import datetime
import base64




plt.rcParams.update({'figure.max_open_warning':0})
sns.set_theme(style="darkgrid")

#parse datetime columns
def date_parser_v1(df, date_cols):
    for feat in date_cols:
        try:
            df[feat +'_year'] = df[feat].dt.year
            df[feat +'_day'] = df[feat].dt.day
            df[feat +'_month'] = df[feat].dt.month
            df[feat +'_hr'] = df[feat].dt.hour
            df[feat +'_min'] = df[feat].dt.minute
            df[feat +'_secs'] = df[feat].dt.second
        except Exception as e:
            st.write("DATE EXCEPTION: ", str(e))
        else:   
            df.drop(columns=feat, axis=1, inplace=True)


#catch columns with 'date' in its name; one of the few methods\
#to catch datetime columns in our dataframe
def date_catcher(dataframe):
    cols = [col for col in dataframe.columns if 'date' in col.lower()]
    return cols


def id_catcher(dataframe):
    df_cols = dataframe.columns
    cols = [col if "id" in col.lower() else df_cols[0] for col in df_cols]
    return cols


# @st.cache(suppress_st_warning=True)
def check_relationship(cols, target, dataframe):
    #plot first 15 features that meets the condition
    df_shape = dataframe.shape[0]
    if df_shape >= 1000:
        n = 1000/5 #divide n by 4 and plot if it meets the condition
    else:
        n = 800/5
    for feat in cols[:15]:
        feat_target_plot, r_ax = plt.subplots()
        #do not plot target against target or IDs against target(too many unique values)
        if feat.lower() == target.lower() or "id" in feat.lower() or len(set(dataframe[feat])) >= int(n/4) or dataframe[feat].is_unique or dataframe[feat].is_monotonic:
            continue
        else:
            sns.barplot(data = dataframe, x=feat, y=target, ax=r_ax)
            r_ax.set_xticklabels(r_ax.get_xticklabels(), rotation=90, fontsize=6)
            st.pyplot(feat_target_plot)


def remove_features(dataframe, cols):
    cols = list(cols)
    dataframe = dataframe.drop(cols, axis=1)
    return dataframe




#remove features that are unique or monotonic
@st.cache(suppress_st_warning=True)
def remove_mono_unique(dataframe, cols):
    cols = list(cols)
    for col in cols:
        if dataframe[col].is_unique or dataframe[col].is_monotonic:
            dataframe = dataframe.drop(col, axis=1)
        else:
            continue
    return dataframe


#set parameter
def model_parameter(classifier):
    param = dict()
    if classifier == "CATBOOST":
        lr = st.sidebar.slider('LEARNING_RATE', 0.01, 1.0, step=0.1)
        eval_metric = st.sidebar.selectbox("EVAL_METRIC", ["F1", "AUC"])
        param["eval_metric"] = eval_metric
        param['learning_rate'] = lr
    if classifier == "SVM":
        C = st.sidebar.slider('C', 0.001, 10.0, step=0.1)
        param['C'] = C
    if classifier == "RANDOMFOREST":
        depth = st.sidebar.slider('MAX_DEPTH', 1, 40, step=1)
        param['n_jobs'] = -1
        param['max_depth'] = depth
    if classifier == "XGBOOST":
        depth = st.sidebar.slider('MAX_DEPTH', 1, 40, step=1)
        param['n_jobs'] = -1
        param['max_depth'] = depth

    return param



def build_model(classifier, params, seed):
    clf = None
    if classifier == "CATBOOST":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(learning_rate=params['learning_rate'],\
            random_state=seed, eval_metric=params["eval_metric"], silent=True)
    if classifier == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=params['K'], random_state=seed)
    if classifier == "RANDOMFOREST":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed)
    if classifier == "XGBOOST":
        from xgboost import XGBClassifier
        clf = XGBClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed)
    
    return clf


def download_csv(dataframe, name, info):
    csv_file = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    return f'<a href="data:file/csv_file;base64,{b64}" download="{name}">{info}</a>'



def main():
    st.title('MACHINE LEARNING FOR YOU..')
    options = ['WELCOME', 'EXPLORE']
    option = st.sidebar.selectbox('Select option: ', options)

    if option == options[0]:
        pass
    elif option == options[1]:
        try:
            train_df = st.file_uploader("Upload Train dataset: ", type=['csv','xlsx'])
            test_df = st.file_uploader("Upload Test dataset: ", type=['csv','xlsx'])
        except Exception as e:
            st.warning(e)
        if train_df is not None and test_df is not None:
            st.success('Upload complete. Status: SUCCESS')
            train = pd.read_csv(train_df)
            test = pd.read_csv(test_df)
            train.columns = map(str.lower, train.columns)
            test.columns = map(str.lower, test.columns)
            train["marker"] = "train"
            test["marker"] = "test"
            df = pd.concat([train, test], axis=0)
            df = df.loc[:, ~df.columns.duplicated()].drop_duplicates()
            keep_cols = df.columns
            datetime_ = st.multiselect('SELECT FEATURES OF TYPE DATE: ', df.columns.tolist(), date_catcher(df))
            datetime_ = list(datetime_)
            if datetime_:
                for col_ in datetime_:
                    try:
                        df[col_] = pd.to_datetime(df[col_], infer_datetime_format=True, format="%y%m%d")
                    except Exception as e:
                        st.write("EXCEPTION (can be ignored): ", str(e))
                    # else:
                st.write("DATETIME COLUMNS PARSED SUCCESSFULLY.")
            else:
                st.write("NO DATE COLUMN FOUND.")


            full_df = None
            full_train = None
            full_test = None


            st.dataframe(df.head(50))
            st.write("SHAPE: ", df.shape)

            id_ = st.multiselect('SELECT *ONE* FEATURE FOR FINAL TEST FILE (ex: ID): ', test.columns.tolist(), ["id" if "id" in test.columns else test.columns.tolist()[0]])
            test_id = test[id_] #store ID for test dataframe
            train_data = df[df["marker"] == "train"]
            # test_data = df[df["marker"] == "test"]
            target_col = st.multiselect("Choose preferred target column: ", train.columns.tolist(), ["target" if "target" in train.columns else train.columns.tolist()[-1]])

            st.write(target_col)
            if target_col:
                target_col = list(target_col)
                target_cp, ax = plt.subplots()
                sns.countplot(data = train_data, x=target_col[0])
                st.pyplot(target_cp)
            else:
                st.warning("TARGET VARIABLE NOT YET DECLARED")
            st.write("INITIALIZING DATE FEATURE ENGINEERING VIA SANGO SHRINE....")
            date_parser_v1(df, datetime_)
            df = df.apply(lambda col: col.str.lower() if (col.dtype == 'object') else col)
            st.dataframe(df)
            st.write("DATE FEATURE ENGINEERING COMPLETE")
            num_df = df.select_dtypes(include=[np.number]).shape[1]
            obj_df = df.select_dtypes(include='object').shape[1]
            if num_df:
                st.write('Numerical column count: ', num_df)
                st.code('''df.select_dtypes(include=[np.number]).shape''', language='python')
            if obj_df:
                cat_cols = [col for col in df.columns if col not in list(df.select_dtypes(include=[np.number]))]
                st.write('Categorical column count: ', obj_df)
                st.code(
                '''#see categorical columns
df.select_dtypes(include=['object']).columns
                ''', language='python')
                st.write(cat_cols[:5])
            st.subheader("Data Summary")
            st.write(df.describe().T)

            train_data = df[df["marker"] == "train"]
            test_data = df[df["marker"] == "test"]

            train_data = train_data.dropna(subset=[target_col[0]])
            test_data[target_col[0]].fillna(value="N/A", inplace=True) #
            pre_miss_df = pd.concat([train_data, test_data], axis=0)
            target_var = train_data[target_col[0]]
            missing_df = pd.DataFrame(data=np.round((pre_miss_df.isnull().sum()/pre_miss_df.shape[0])*100,1), columns=["missing (%)"])
            st.dataframe(missing_df.T)
            if missing_df["missing (%)"].any(): #check for nans (True if any) 
                keep = st.slider("KEEP COLUMNS WITH MISSING DATA (%)", 0, 100, 50, 10)
            
                keep_cols = missing_df[missing_df["missing (%)"] <= int(keep)].index
                keep_cols = list(keep_cols)

              
                handle_nan = st.selectbox(label="HANDLE NANs", options=["MODE","MEDIAN", "MEAN"])
                if handle_nan == "MODE":
                    full_train = train_data[keep_cols].fillna(train_data[keep_cols].mode().iloc[0])
                    full_test = test_data[keep_cols].fillna(test_data[keep_cols].mode().iloc[0])

                elif handle_nan == "MEDIAN":
                    full_train = train_data[keep_cols].fillna(train_data[keep_cols].median().iloc[0])
                    full_test = test_data[keep_cols].fillna(test_data[keep_cols].median().iloc[0])
     
                elif handle_nan == "MEAN":
                    full_train = train_data[keep_cols].fillna(train_data[keep_cols].mean().iloc[0])
                    full_test = test_data[keep_cols].fillna(test_data[keep_cols].mean().iloc[0])
               
                else:
                    st.write("NO SELECTED WAY TO HANDLE NAN") #precaution


                st.write("MISSING DATA PADDED")
            else:
                st.write("NO MISSING DATA")

            if full_train is not None and full_test is not None:
                new_df = pd.concat([full_train, full_test], axis=0) #use padded data
            else:
                new_df = pre_miss_df #use this since missing data wasn't present
            st.dataframe(new_df.head(50))
            st.write("SHAPE: ", new_df.shape)
            if new_df.shape[1] > 50:
                st.write("ABSOLUTE CORRELATION WITH TARGET VARIABLE")
                st.write(full_train.corr()[target_col[0]].sort_values(by=target_col[0], ascending=False).T)
                st.write("[correlation is not causation]")
            else:
                heatmap_fig, ax=plt.subplots()
                sns.heatmap(full_train.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
                st.pyplot(heatmap_fig)
            

            new_df_cols = list(new_df.columns)
            if target_col[0] in list(new_df.columns):
                new_df_cols.remove(target_col[0])
            if id_[0] in list(new_df.columns):
                new_df_cols.remove(id_[0])


            st.subheader("PLOTTING POSSIBLE RELATIONSHIP WITH TARGET FEATURE")
            check_relationship(new_df_cols, target_col[0], full_train)

            #handle features excluded
            remove_feat = st.multiselect("SELECT FEATURE(S) TO DROP", new_df_cols)
            if remove_feat:
                new_df = remove_features(new_df, remove_feat)
            else:
                st.write("KEEPING ALL FEATURES")

            st.dataframe(new_df.head(50))
            st.write(new_df.shape)

            #test_id = new_df[new_df["marker"] == "test"][id_] #store ID for test dataframe

            #remove monotonic or unique features
            new_df = remove_mono_unique(dataframe=new_df, cols=new_df.columns)
            st.dataframe(new_df.head(50))
            st.write(new_df.shape)
            st.write("MONOTONIC AND UNIQUE FEATURES REMOVED")


            not_dummy = [target_col[0], "target", "marker", "claim", "pred"]
            exclude_cols = [col for col in new_df.columns if col not in not_dummy]
            exclude_cols = list(set(exclude_cols).intersection(list(new_df.select_dtypes(include='object').columns)))
            dum_df = pd.get_dummies(new_df, columns=exclude_cols, drop_first=True)
            st.dataframe(dum_df.head(100))
            st.write(dum_df.shape)
            st.write("CATEGORICAL FEATURES ENCODED")


            dum_train = dum_df[dum_df["marker"] == "train"].drop([target_col[0], "marker"], axis=1)
            dum_train_y = dum_df[dum_df["marker"] == "train"][target_col[0]].astype(int)
            dum_test = dum_df[dum_df["marker"] == "test"].drop([target_col[0], "marker"], axis=1)
        

            scaler = ["STANDARDSCALER", "MIN-MAX SCALER"]
            scaler_option = st.selectbox("SCALE DATA USING: ", scaler)
            if scaler_option == scaler[0]:
                from sklearn.preprocessing import StandardScaler
                ss = StandardScaler()
                Xtrain = pd.DataFrame(ss.fit_transform(dum_train), columns=dum_train.columns)
                test = pd.DataFrame(ss.transform(dum_test), columns=dum_test.columns)
            elif scaler_option == scaler[1]:
                from sklearn.preprocessing import MinMaxScaler
                mm = MinMaxScaler()
                Xtrain = pd.DataFrame(mm.fit_transform(dum_train), columns=dum_train.columns)
                test = pd.DataFrame(mm.transform(dum_test), columns=dum_test.columns)
            else:
                st.write("NO SCALER SELECTED")

            st.subheader("Train Data")
            
            st.dataframe(Xtrain.head(1000))
            st.write(Xtrain.shape)
            st.markdown(download_csv(Xtrain, "cpt_train.csv", info="DOWNLOAD TRAIN FILE"), unsafe_allow_html=True)
            
            st.subheader("Test Data")
            
            st.dataframe(test.head(1000))
            st.write(test.shape)
            st.markdown(download_csv(test, "cpt_test.csv", info="DOWNLOAD TEST FILE"), unsafe_allow_html=True)

            st.header('TRAINING/TESTING SECTION')

            models = ['CATBOOST', 'KNN', 'RANDOMFOREST', 'XGBOOST']
            model = st.sidebar.selectbox('Select option: ', models)
            seed = st.sidebar.slider('SEED', 1, 50, step=1)

            params = model_parameter(model)
            model_ = build_model(model, params, seed)

            #ensuring producibility
            np.random.seed(seed=seed)
            np.random.RandomState(seed=seed)

            X_train, X_test, y_train, y_test = train_test_split(Xtrain, dum_train_y, test_size=.4, random_state=seed)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.7, random_state=seed)
            st.write("BUILDING MODEL WITH: ", model , model_.get_params())
            st.write("TRAIN-VAL-TEST SPLIT: 60%:30%:10%")
            st.write(X_train.shape, X_val.shape, X_test.shape)
            model_.fit(X_train, y_train)
            y_val_ = model_.predict(X_val)
            y_test_ = model_.predict(X_test)
            st.write("VALIDATION PARTITION REPORT")
            accuracy_val = metrics.classification_report(y_val_, y_val)
            st.write(accuracy_val)
            st.write("TEST PARTITION REPORT")
            accuracy_test = metrics.classification_report(y_test_, y_test)
            st.write(accuracy_test)

            st.write("TEST ACCURACY: ", metrics.accuracy_score(y_test_, y_test))
            st.write("TEST F1 SCORE: ", metrics.f1_score(y_test_, y_test))

            test_id[target_col[0]] =  model_.predict(test)
            if test_id is not None:
                # pred_dict = {"ID": np.array(test_id.values), target_col[0]: model_.predict(test)}
                # test_pred = pd.DataFrame.from_dict(pred_dict)
                st.write(test_id.head(1000))
                st.write(test_id.shape)
                st.write("")
                st.write("MODEL ESTABLISHED. YAY!")
                st.balloons()
            else:
                st.write("YOUR MODEL FAILED YTO COMPLETE")
            

        elif train_df:
            st.write("YOU NEED TEST DATASET TOO")
        elif test_df:
            st.write("YOU NEED TRAIN DATASET AS WELL")
        else:
            st.write("ABEG UPLOAD TRAIN AND TEST DATASET")
    else:
        st.write('INVALID ARGUMENT! ')


if __name__ == "__main__":
    main()