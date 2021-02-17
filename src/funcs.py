#streamlit
import streamlit as st

#computational and dataframe
import pandas as pd
import numpy as np
import numbers

#plots
import matplotlib.pyplot as plt
import seaborn as sns
# import altair as alt

#sklearn - data, metrics and algos
from sklearn import datasets #tests
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


#
import imblearn

import sklearn
import sys

import datetime
import base64
import urllib
 

#borrowed from kaggle(@arjanso)
@st.cache(suppress_st_warning=True)
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    NAlist = []
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
    mem_usg = props.memory_usage().sum() / 1024**2 
    mem_reduced = np.round(100*mem_usg/start_mem_usg, 2)
    return props, mem_reduced



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
    #plot first 20 features that meets the condition
    df_shape = dataframe.shape[0]
    if df_shape >= 1000:
        n = 1000/5 #divide n by 4 and plot if it meets the condition
    else:
        n = 800/5
    for feat in cols[:20]:
        feat_target_plot, r_ax = plt.subplots()
        #do not plot target against target or IDs against target(too many unique values)
        if feat.lower() == target.lower() or "id" in feat.lower() or feat == "marker" or len(set(dataframe[feat])) >= int(n/4) or dataframe[feat].is_unique or dataframe[feat].is_monotonic:
            continue
        else:
            try:
                sns.barplot(data = dataframe, x=feat, y=target, ax=r_ax)
                r_ax.set_xticklabels(r_ax.get_xticklabels(), rotation=90, fontsize=6)
            except Exception as e:
                st.write("EXCEPTION: ", str(e))
            else:
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


#feature scaling
def feature_scaling(train, test):
    SCALER = ["STANDARDSCALER", "MIN-MAX SCALER"]
    scaler_option = st.selectbox("SCALE DATA USING: ", SCALER)
    if scaler_option == SCALER[0]:
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        train_scaled = pd.DataFrame(ss.fit_transform(train), columns=train.columns)
        test_scaled = pd.DataFrame(ss.transform(test), columns=test.columns)
    elif scaler_option == SCALER[1]:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        train_scaled = pd.DataFrame(mm.fit_transform(train), columns=train.columns)
        test_scaled = pd.DataFrame(mm.transform(test), columns=test.columns)
    else:
        st.write("NO SCALER METHOD SELECTED")
    return train_scaled, test_scaled


#set parameter
def model_parameter(classifier):
    param = dict()
    if classifier == "CATBOOST":
        lr = st.sidebar.slider('LEARNING_RATE', 0.01, 1.0, step=0.1)
        eval_metric = st.sidebar.selectbox("EVAL_METRIC", ["F1", "AUC"])
        param["eval_metric"] = eval_metric
        param['learning_rate'] = lr
    if classifier == "KNN":
        K = st.sidebar.slider('n_neighbor', 1, 10, step=1)
        param['K'] = K
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
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    if classifier == "RANDOMFOREST":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed)
    if classifier == "XGBOOST":
        from xgboost import XGBClassifier
        clf = XGBClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed)
    
    return clf


@st.cache(suppress_st_warning=True)
def initialize_model(model, Xtrain_file, ytrain_file, test_file, test_dataframe, target_var_, seed):
    X_train, X_test, y_train, y_test = train_test_split(Xtrain_file, ytrain_file, test_size=.4, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.7, random_state=seed)
    st.write("BUILDING MODEL WITH: ", model , model.get_params())
    st.write("TRAIN-VAL-TEST SPLIT: 60%:30%:10%")
    st.write(X_train.shape, X_val.shape, X_test.shape)
    model.fit(X_train, y_train)
    # y_train_ = model.predict(X_train, y_train)
    y_val_ = model.predict(X_val)
    y_test_ = model.predict(X_test)
    st.write("VALIDATION PARTITION REPORT")
    accuracy_val = sklearn.metrics.classification_report(y_val_, y_val)
    st.write(accuracy_val)
    st.write("TEST PARTITION REPORT")
    accuracy_test = sklearn.metrics.classification_report(y_test_, y_test)
    st.write(accuracy_test)
    train_ = 0 #fix
    val_ = (y_val_, y_val)
    test_ = (y_test_, y_test)
    test_dataframe[target_var_] =  model.predict(test_file)

    return train_, val_, test_, test_dataframe



def balance_out(train, target, seed):
    BALANCE_OPTS = ["DEFAULT","SMOTE", "RANDOM OVERSAMPLER", "RANDOM UNDERSAMPLER"]
    balance_type = st.selectbox("DOWNSAMPLE/UPSAMPLE YOUR DATA: ", BALANCE_OPTS)

    if balance_type == BALANCE_OPTS[0]:
        return train, target, balance_type
    elif balance_type == BALANCE_OPTS[1]:
        smote = imblearn.over_sampling.SMOTE(random_state=seed, n_jobs=-1)
        new_train, new_target = smote.fit_resample(train, target["target"])
        new_train = pd.DataFrame(new_train, columns=train.columns)
        new_target = pd.DataFrame(new_target, columns=["target"])
        st.write(f"Upsampled using SMOTE. \nTrain set (class 1): ", new_target[new_target["target"] == 1].shape[0], "\nTrain set (class 0): ", new_target[new_target["target"] == 0].shape[0])
        return new_train, new_target, balance_type
    elif balance_type == BALANCE_OPTS[2]:
        randomover = imblearn.over_sampling.RandomOverSampler(random_state=seed)
        new_train, new_target = randomover.fit_resample(train, target["target"])
        new_train = pd.DataFrame(new_train, columns=train.columns)
        new_target = pd.DataFrame(new_target, columns=["target"])
        st.write(f"Upsampled using RANDOMOVERSAMPLER. \nTrain set (class 1): ", new_target[new_target["target"] == 1].shape[0], "\nTrain set (class 0): ", new_target[new_target["target"] == 0].shape[0])
        return new_train, new_target, balance_type
    else: # balance_type == BALANCE_OPTS[3]
        randomunder = imblearn.under_sampling.RandomUnderSampler(random_state=seed)
        new_train, new_target = randomunder.fit_resample(train, target["target"])
        new_train = pd.DataFrame(new_train, columns=train.columns)
        new_target = pd.DataFrame(new_target, columns=["target"])
        st.write(f"Downsampled using RandomUnderSampler. \nTrain set (class 1): ", new_target[new_target["target"] == 1].shape[0], "\nTrain set (class 0): ", new_target[new_target["target"] == 0].shape[0])
        return new_train, new_target, balance_type
    

def download_csv(dataframe, name, info):
    csv_file = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    return f'<a href="data:file/csv_file;base64,{b64}" download="{name}">{info}</a>'


@st.cache(suppress_st_warning=True)
def get_content(path):
    resp = ''
    url = 'https://raw.githubusercontent.com/iameo/ml-cpt/master/' + path
    try:
        resp = urllib.request.urlopen(url)
    except Exception as e:
        return f'README requires connectivity (or cache)! But do proceed to Explore.'
    return resp.read().decode("utf-8")
