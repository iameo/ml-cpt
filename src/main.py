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




plt.xticks(rotation=45)
sns.set_theme(style="darkgrid")

#parse datetime columns
def date_parser_v1(df,date_cols):
    for feat in date_cols:
        try:
            df[feat +'_year'] = df[feat].dt.year
            df[feat +'_day'] = df[feat].dt.day
            df[feat +'_month'] = df[feat].dt.month
            df[feat +'_hr'] = df[feat].dt.hour
            df[feat +'_min'] = df[feat].dt.mins
            df[feat +'_secs'] = df[feat].dt.seconds
        except Exception as e:
            st.write("DATE EXCEPTION: ", str(e))
        else:   
            df.drop(columns=feat, axis=1, inplace=True)


#catch columns with 'date' in its name; one of the few methods\
#to catch datetime columns in our dataframe
def date_catcher(dataframe):
    cols = [col for col in dataframe.columns if 'date' in col.lower()]
    return cols

# def date_parser(dataframe, date_ = False, time_ = False):
#     datetime_column = [col for col in dataframe.columns if isinstance(col, datetime.datetime)]
#     new_dataframe = None
#     new_dataframe['day'] = ''
#     new_dataframe['month'] = ''
#     new_dataframe['year'] = ''

#     return new_dataframe


def check_relationship(cols, target, dataframe):
    for feat in cols:
        feat_target_plot, r_ax = plt.subplots()
        #do not plot target against target or IDs against target(too many unique values)
        if feat.lower() == target.lower() or "id" in feat.lower() or len(set(dataframe[feat])) >= 40:
            continue
        else:
            sns.barplot(data = dataframe, x=feat, y=target, ax=r_ax)
            r_ax.set_xticklabels(r_ax.get_xticklabels(), rotation=90, fontsize=6)
            st.pyplot(feat_target_plot)


def remove_features(dataframe, cols):
    cols = list(cols)
    dataframe.drop(cols, axis=1, inplace=True)
    return dataframe


#remove features that are unique or monotonic
def remove_big_unique(dataframe, cols):
    cols = list(cols)
    for col in cols:
        if dataframe[col].is_unique or dataframe[col].is_monotonic:
            dataframe.drop(col, axis=1, inplace=True)
        else:
            continue

#set parameter
def model_parameter(classifier):
    param = dict()
    if classifier == "CATBOOST":
        lr = st.sidebar.slider('LEARNING_RATE', 0.01, 1.0)
        eval_metric = st.sidebar.selectbox("EVAL_METRIC", ["F1", "AUC"])
        param["eval_metric"] = eval_metric
        param['learning_rate'] = lr
        param["verbose"] = True
    if classifier == "SVM":
        C = st.sidebar.slider('C', 0.001, 10.0)
        param['C'] = C
    if classifier == "RANDOMFOREST":
        depth = st.sidebar.slider('MAX_DEPTH', 1, 40)
        param['n_jobs'] = -1
        param['max_depth'] = depth
        param["verbose"] = 1
    if classifier == "XGBOOST":
        depth = st.sidebar.slider('MAX_DEPTH', 1, 40)
        param['n_jobs'] = -1
        param['max_depth'] = depth
        param["verbose"] = 1

    return param
    
def build_model(classifier, params, seed):
    clf = None
    if classifier == "CATBOOST":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(learning_rate=params['learning_rate'],\
            random_state=seed, eval_metric=params["eval_metric"], verbose=params["verbose"])
    if classifier == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=params['K'], random_state=seed)
    if classifier == "RANDOMFOREST":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed, verbose=params["verbose"])
    if classifier == "XGBOOST":
        from xgboost import XGBClassifier
        clf = XGBoostClassifier(max_depth=params['max_depth'],\
            n_jobs=params["n_jobs"], random_state=seed, verbose=params["verbose"])
    
    return clf

@st.cache(suppress_st_warning=True)
def main():
    st.title('MACHINE LEARNING FOR YOU..')
    options = ['DEFAULT', 'CUSTOM']
    option = st.sidebar.selectbox('Select option: ', options)

    if option == 'DEFAULT':
        pass
    elif option == 'CUSTOM':
        try:
            user_data = st.file_uploader("Upload dataset: ", type=['csv','xlsx'])
        except Exception as e:
            st.warning(e)
        if user_data is not None:
            st.success('Upload complete. Status: SUCCESS')
            df = pd.read_csv(user_data)
            keep_cols = df.columns
            datetime_ = st.multiselect('SELECT FEATURES OF TYPE DATE: ', df.columns.tolist(), date_catcher(df))
            datetime_ = list(datetime_)
            if datetime_:
                for col_ in datetime_:
                    try:
                        df[col_] = pd.to_datetime(df[col_], infer_datetime_format=True, format="%y%m%d")
                        # st.write("done")
                    except Exception as e:
                        st.write("EXCEPTION (can be ignored): ", str(e))
                    # else:
                    #     st.write("DATETIME PARSED SUCCESSFULLY!")
            else:
                st.write("NO DATE COLUMN FOUND.")
            full_df = None
            st.dataframe(df.head(10))
            st.write("SHAPE: ", df.shape)
            if "marker" not in df.columns:
                st.write("EXCEPTION (can be ignored): NO MARKER COLUMN INDICATED FOR TARGET VARIABLE")
            target_col = st.multiselect("Choose preferred target column: ", df.columns.tolist(), ["target" if "target" in df.columns else df.columns.tolist()[-1]])
            # st.write(["target" if "target" in df.columns else df.columns.tolist()[-1]])
            
            if target_col:
                target_col = list(target_col)
                st.write(target_col)
                target_cp, ax = plt.subplots()
                sns.countplot(data = df, x=target_col[0])
                st.pyplot(target_cp)
            else:
                st.warning("TARGET VARIABLE NOT YET DECLARED")
            st.write("INITIALIZING DATE FEATURE ENGINEERING VIA SANGO SHRINE....")
            date_parser_v1(df, datetime_)
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
                st.write(list(df.select_dtypes(include='object'))[:5], "\n[you may want to encode]")
            st.subheader("Data Summary")
            st.write(df.describe().T)
            pre_miss_df = df.copy()
            target_var = df[target_col[0]]
            missing_df = pd.DataFrame(data=np.round((df.isnull().sum()/df.shape[0])*100,1), columns=["missing (%)"])
            if missing_df["missing (%)"].any(): #check for nans (True if any)
                st.dataframe(missing_df.T)
                keep = st.number_input("KEEP COLUMNS WITH MISSING DATA (%)", 50)
                if isinstance(keep, numbers.Number):
                    keep_cols = missing_df[missing_df["missing (%)"] <= keep].index
                    keep_cols = list(keep_cols)
                    keep_cols.remove(str(target_col[0]))

                else:
                    st.warning("VALUE MUST BE AN INTEGER")
                
                handle_nan = st.selectbox(label="HANDLE NANs", options=["MODE","MEDIAN", "MEAN"])
                if handle_nan == "MODE":
                    full_df = df[keep_cols].fillna(df[keep_cols].mode().iloc[0])
                elif handle_nan == "MEDIAN":
                    full_df = df[keep_cols].fillna(df[keep_cols].median().iloc[0])
                elif handle_nan == "MEAN":
                    full_df = df[keep_cols].fillna(df[keep_cols].mean().iloc[0])
                else:
                    st.write("NO SELECTED WAY TO HANDLE NAN")
                # fillNans = st.number_input("FILL NANS WITH ", -999)
                # if fillNans:
                #     st.write(type(fillNans))
                #     full_df = df[keep_cols].fillna(fillNans)
                #     st.dataframe(full_df)
                #     st.write("NANS UPDATED WITH VALUE: {fillNans}".format(fillNans=fillNans))
                # else:
                #     # full_df = None
                #     pass

                st.write("MISSING DATA ELIMINATED!")
            else:
                st.write("NO MISSING DATA")

            if full_df is not None:
                new_df = full_df
            else:
                new_df = pre_miss_df
            st.dataframe(new_df.head(20))
            st.write("SHAPE: ", new_df.shape)
            if new_df.shape[1] > 50:
                st.write("ABSOLUTE CORRELATION WITH TARGET VARIABLE")
                st.write(new_df.corr()[target_col[0]].sort_values(by=target_col[0], ascending=False).T)
                st.write("[correlation is not causation]")
            else:
                heatmap_fig, ax=plt.subplots()
                sns.heatmap(new_df.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
                st.pyplot(heatmap_fig)
            
            # st.write(type(target_col[0]))
            # check_relationship(cols=cols[:10], target=target_col[0], dataframe=new_df)
            # target_var = new_df[target_col[0]]
            new_df_cols = list(new_df.columns)
            if target_col[0] in list(new_df.columns):
                new_df_cols.remove(target_col[0])
            #handle features excluded
            remove_feat = st.multiselect("SELECT FEATURE(S) TO DROP", new_df_cols)
            if remove_feat:
                new_df = remove_features(new_df, remove_feat)
            else:
                st.write("KEEPING ALL FEATURES")
            st.dataframe(new_df)
            st.write(new_df.shape)

            remove_big_unique(dataframe=new_df, cols=new_df.columns)
            st.dataframe(new_df)
            st.write("MONOTONIC AND UNIQUE FEATURES REMOVED")
            st.write(new_df.shape)

            dum_df = pd.get_dummies(new_df, drop_first=True)
            st.dataframe(dum_df)
            st.write("CATEGORICAL FEATURES ENCODED")
            st.write(dum_df.shape)

            # st.dataframe(pd.get_dummies(new_df, drop_first=True))
            # st.dataframe(pd.get_dummies(new_df))
            # new_df.drop(target_var, axis=1, inplace=True)
            
            scaler = ["STANDARDSCALER", "MIN-MAX SCALER"]
            scaler_option = st.selectbox("SCALE DATA USING: ", scaler)
            if scaler_option == scaler[0]:
                from sklearn.preprocessing import StandardScaler
                ss = StandardScaler()
                Xtrain = pd.DataFrame(ss.fit_transform(dum_df), columns=dum_df.columns)
            elif scaler_option == scaler[1]:
                from sklearn.preprocessing import MinMaxScaler
                mm = MinMaxScaler()
                Xtrain = pd.DataFrame(mm.fit_transform(dum_df), columns=dum_df.columns)
            else:
                st.write("NO SCALER SELECTED")
            st.dataframe(Xtrain)
            st.title('MACHINE LEARNING FOR YOU..')
            models = ['CATBOOST', 'KNN', 'RANDOMFOREST', 'XGBOOST']
            model = st.sidebar.selectbox('Select option: ', models)
            seed = st.sidebar.slider('SEED', 1, 300)

            params = model_parameter(model)
            model_ = build_model(model, params, seed)

            X_train, X_test, y_train, y_test = train_test_split(Xtrain, target_var, test_size=.3, random_state=seed)

            st.write("BUILDING MODEL WITH: ", model , model_.get_params())
            model_.fit(X_train, y_train)
            y_pred = model_.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(accuracy)
            

    else:
        st.write('INVALID ARGUMENT! ')

# st.title('ML FOR BEGINNERS')

# dataset_ = st.sidebar.selectbox('Select dataset', ('Iris', 'Breast'))
# task_ = st.sidebar.selectbox('select task', ('EDA', 'TRAIN-TEST'))
# classifier_ = st.sidebar.selectbox('Select Algorithm', ('CATBOOST', 'KNN', 'SVM', 'LOG. REGRESSION'))


# def fetch_dataset(name):
#     data = None
#     if name == 'Iris':
#         data = datasets.load_iris()
#     elif name == 'Breast':
#         print('breast')
#         data = datasets.load_breast_cancer()
#     else:
#         st.write("Data can not be None!")
#     x = data.data
#     y = data.target
#     return x,y

# x, y = fetch_dataset(dataset_)
# st.dataframe(x)
# st.write('SHAPE: ', x.shape)

# target_len = len(np.unique(y))
# if target_len <= 20: #test case - classification
#     st.write('ML TYPE - SUPERVISED CLASSIFICATION TASK')
# else:
#     st.write('ML TYPE - SUPERVISED REGRESSION TASK')

# st.title('EDA')
# num_df = pd.DataFrame(x).select_dtypes(include=[np.number]).shape[1]
# obj_df = pd.DataFrame(x).select_dtypes(include='object').shape[1]
# if num_df:
#     st.write('Numerical column count: ', num_df)
#     st.code('''pd.DataFrame(x).select_dtypes(include='float').shape''', language='python')
# if obj_df:
#     st.write('Categorical column count: ', obj_df)
#     st.code('''pd.DataFrame(x).select_dtypes(include='object').shape''', language='python')
#     st.write('(you may want to encode the objects)')

# boxplot_fig = plt.figure()
# sns.boxplot(data = x, orient='h')
# st.pyplot(boxplot_fig)

# boxplot_code = '''plt.figure(); sns.boxplot(data=x, orient='h')'''
# st.code(boxplot_code, language='python')


# hist_fig, ax = plt.subplots()
# ax.hist(x, bins=20)
# st.pyplot(hist_fig)

# with st.echo():
#     df = pd.DataFrame(x)
# st.write('echo')



if __name__ == "__main__":
    main()