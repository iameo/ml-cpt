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


import sklearn
import sys

import datetime


from funcs import (get_content, reduce_mem_usage, date_catcher, date_parser_v1, check_relationship, remove_features,\
     remove_mono_unique, download_csv, model_parameter, build_model, initialize_model, download_csv, feature_scaling,\
         balance_out)

from raw_code import df_head, df_shape, plot_target, heatmap_code, heatmap_sns


pd.options.mode.chained_assignment = None

plt.rcParams.update({'figure.max_open_warning':0})
sns.set_theme(style="darkgrid")


MODELS = ['CATBOOST', 'KNN', 'RANDOMFOREST', 'XGBOOST']
SCALER = ["STANDARDSCALER", "MIN-MAX SCALER"]

def date_month():
    from datetime import datetime
    today = datetime.now()
    return today.month

def on_completion():
    month_today = date_month()
    st.write(month_today)

    if month_today in [10,12,1,2,3]:
        st.snow() #not adapted
    else:
        st.balloons()




def main():
    st.title('MACHINE LEARNING FOR YOU...')

    
    options = ['WELCOME', 'EXPLORE']
    option = st.sidebar.selectbox('Select option: ', options)

    if option == options[0]:
        welcome_text = st.markdown(get_content("README.md"))
    elif option == options[1]:
        
        #ensuring producibility
        seed = st.sidebar.slider('SEED', 1, 50, step=1)

        np.random.seed(seed=seed)
        # np.random.RandomState(seed=seed)
        
        # welcome_text.empty()
        try:
            train_df = st.file_uploader("Upload Train dataset: ", type=['csv','xlsx'])
            test_df = st.file_uploader("Upload Test dataset: ", type=['csv','xlsx'])
        except Exception as e:
            st.warning(e)
        if train_df and test_df:
            # ##st.code("""
            # df.select_dtypes(include=[np.number]).shape
            # """, language='python')
            st.success('Upload complete. Status: SUCCESS')

            #train, test data loading...
            train, test = pd.read_csv(train_df), pd.read_csv(test_df)

            train.columns = map(str.lower, train.columns)
            test.columns = map(str.lower, test.columns)
            train["marker"] = "train"
            test["marker"] = "test"
            df = pd.concat([train, test], axis=0)
            df, mem_reduced = reduce_mem_usage(df)
            st.write("MEMORY SAVED: ", mem_reduced,"MB")
            df = df.loc[:, ~df.columns.duplicated()].drop_duplicates()
            keep_cols = df.columns
            datetime_ = st.multiselect('SELECT FEATURES OF TYPE DATE: ', df.columns.tolist(), date_catcher(df))
            if datetime_:
                datetime_ = list(datetime_)
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


            st.dataframe(df)

            #show code
            df_head()

            st.write("SHAPE: ", df.shape)

            #show code
            df_shape()

            id_ = st.multiselect('SELECT *ONE* FEATURE FOR SUBMISSION FILE (ex: ID): ', test.columns.tolist(), ["id" if "id" in test.columns else test.columns.tolist()[0]])
            if not id_:
                st.warning("YOU REALLY SHOULD PICK AN IDENTIFIER FOR YOUR TEST SUBMISSION FILE.")
            test_id = test[id_] #store ID for test dataframe
            train_data = df[df["marker"] == "train"]

            target_col = st.multiselect("Choose preferred target column: ", train.columns.tolist(), ["target" if "target" in train.columns else train.columns.tolist()[-1]])

            if target_col:
                target_col = list(target_col)
                target_cp, ax = plt.subplots()
                sns.countplot(data = train_data, x=target_col[0])
                st.pyplot(target_cp)
                plot_target()
            else:
                st.warning("TARGET VARIABLE NOT YET DECLARED")
            
            if len(datetime_) < 1:
                st.write("NO DATETIME COLUMN FOUND. SKIPPING......")
            else:
                with st.spinner("INITIALIZING DATE FEATURE ENGINEERING VIA SANGO SHRINE...."):
                    date_parser_v1(df, datetime_)
            
            df = df.apply(lambda col: col.str.lower() if (col.dtype == 'object') else col)

            st.dataframe(df)

            st.write("DATE FEATURE ENGINEERING COMPLETE")
            
            num_df = df.select_dtypes(include=[np.number]).shape[1]
            obj_df = df.select_dtypes(include='object').shape[1]
            if num_df:
                st.write('Numerical column count: ', num_df)
                st.code('''df.select_dtypes(include=[np.number])''', language='python')
            if obj_df:
                cat_cols = [col for col in df.columns if col not in list(df.select_dtypes(include=[np.number]))]
                st.write('Categorical column count: ', obj_df)
                
                #show code
                st.code(
                '''#see categorical columns
df.select_dtypes(include=['object'])
                ''', language='python')

                st.write(cat_cols[:5])
            st.subheader("Data Summary")
            st.write(df.describe().T)

            #show code
            st.code('''
            df.describe()
            ''', language='python')
            
            train_data = df[df["marker"] == "train"]
            test_data = df[df["marker"] == "test"]

            train_data = train_data.dropna(subset=[target_col[0]])
            test_data.loc[test_data["marker"] == "test", target_col[0]] = "N/A" #
            pre_miss_df = pd.concat([train_data, test_data], axis=0)
            target_var = train_data[target_col[0]]
            missing_df = pd.DataFrame(data=np.round((pre_miss_df.isnull().sum()/pre_miss_df.shape[0])*100,1), columns=["missing (%)"])
            
            #show code
            st.code('''
             pd.DataFrame(data=np.round((train.isnull().sum()/train.shape[0])*100,1), columns=["missing (%)"])
             ''', language='python')

            st.dataframe(missing_df.T)
            if missing_df["missing (%)"].any(): #check for nans (True if any) 
                keep = st.slider("KEEP COLUMNS WITH MISSING DATA (%)", 0, 100, 50, 10)
            
                keep_cols = missing_df[missing_df["missing (%)"] <= int(keep)].index
                keep_cols = list(keep_cols)

              
                handle_nan = st.selectbox(label="HANDLE NANs", options=["MODE","MEDIAN", "MEAN"])
                """Read on SimpleImputer"""
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

            #conserve memory
            df = None

            if full_train is not None and full_test is not None:
                try:
                    new_df = pd.concat([full_train, full_test], axis=0) #use padded data
                except Exception as e:
                    st.exception(str(e))
            else:
                new_df = pre_miss_df #use this since missing data wasn't present
            st.dataframe(new_df.head(50))
            st.write("SHAPE: ", new_df.shape)
            if new_df.shape[1] > 50:
                st.write("ABSOLUTE CORRELATION WITH TARGET VARIABLE")
                with st.spinner('Generating Heatmap.....'):
                    st.write(new_df[new_df["marker"] == "train"].corr()[target_col[0]].sort_values(by=target_col[0], ascending=False).T)
                    st.write("[correlation is not causation]")

                    #show code
                heatmap_code()

            else:
                with st.spinner('Generating Heatmap...'):
                    heatmap_fig, ax=plt.subplots()
                    sns.heatmap(new_df[new_df["marker"] == "train"].corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
                    st.pyplot(heatmap_fig)

                #show code
                heatmap_sns()
            

            new_df_cols = list(new_df.columns)
            if target_col[0] in list(new_df.columns):
                new_df_cols.remove(target_col[0])
            if id_[0] in list(new_df.columns):
                new_df_cols.remove(id_[0])


            st.subheader("PLOTTING POSSIBLE RELATIONSHIP WITH TARGET FEATURE")
            check_relationship(new_df_cols, target_col[0], new_df[new_df["marker"] == "train"])

            #handle features excluded
            remove_feat = st.multiselect("SELECT FEATURE(S) TO DROP", new_df_cols)
            if remove_feat:
                new_df = remove_features(new_df, remove_feat)

                #show code
                st.code('''
                df.dropna([list of columns to drop]), axis=1, inplace=True)
                ''', language='python')

            else:
                st.write("KEEPING ALL FEATURES")

            st.dataframe(new_df.head(50))
            st.write(new_df.shape)

            #test_id = new_df[new_df["marker"] == "test"][id_] #store ID for test dataframe

            #remove monotonic or unique features
            with st.spinner('Removing Monotonic or Unique features.....'):
                new_df = remove_mono_unique(dataframe=new_df, cols=new_df.columns)
            st.dataframe(new_df.head(50))
            st.write(new_df.shape)
            st.write("MONOTONIC AND UNIQUE FEATURES REMOVED")


            NOT_DUMMY = [target_col[0], "target", "marker", "claim", "prediction", "response"] #features we do not need the dummy for

            with st.spinner('Getting dummies....'):
                exclude_cols = [col for col in new_df.columns if col not in NOT_DUMMY]
                exclude_cols = list(set(exclude_cols).intersection(list(new_df.columns)))
                dum_df = pd.get_dummies(new_df[exclude_cols], drop_first=True)

            dum_df["marker"] = new_df["marker"].copy()
            dum_df[target_col[0]] = new_df[target_col[0]].copy()
            st.dataframe(dum_df.head(100))
            st.write(dum_df.shape)
            st.write("CATEGORICAL FEATURES ENCODED")

            new_df = None

            dum_train = dum_df[dum_df["marker"] == "train"].drop([target_col[0], "marker"], axis=1)
            st.dataframe(dum_train.head())

            try:
                dum_train_y = pd.DataFrame(list(dum_df.iloc[:dum_train.shape[0]][target_col[0]].astype('int')), columns=["target"])
                dum_test = dum_df[dum_df["marker"] == "test"].drop([target_col[0], "marker"], axis=1)
            except Exception as e:
                st.exception(e)
            
            #feature scaling
            with st.spinner('Feature scaling....'):
                train_scaled, test_scaled = feature_scaling(dum_train, dum_test)
            
            st.subheader("Train Data")
            
            st.dataframe(train_scaled.head(200))
            st.write(train_scaled.shape)
            st.markdown(download_csv(train_scaled, "cpt_train.csv", info="DOWNLOAD TRAIN FILE"), unsafe_allow_html=True)
            
            st.subheader("Test Data")
            
            st.dataframe(test_scaled.head(200))
            st.write(test_scaled.shape)
            st.markdown(download_csv(test_scaled, "cpt_test.csv", info="DOWNLOAD TEST FILE"), unsafe_allow_html=True)

            #downsample/upsample
            # _train = _target = None
            unique_class = len(set(dum_train_y['target']))
            if unique_class < 2:
                st.warning('less than two classes observed. Halting.....')
                st.stop()
            
            BALANCE_OPTS = ["DEFAULT","SMOTE", "RANDOM OVERSAMPLER", "RANDOM UNDERSAMPLER"]
            balance_type = st.selectbox("DOWNSAMPLE/UPSAMPLE YOUR DATA: ", BALANCE_OPTS)
            if unique_class == 2: #binary classification
                classones = int((dum_train_y[dum_train_y["target"] == 1].count()/dum_train_y.shape[0])*100)
                classzeroes = int((dum_train_y[dum_train_y["target"] == 0].count()/dum_train_y.shape[0])*100)
                if classones >= 70 or classzeroes >= 70:
                    st.warning("IMBALANCED TRAINING SET DETECTED!")
                    st.write("CLASS 1(",classones,"%) to CLASS 0(",classzeroes,"%)")
                    with st.spinner(''):
                        try:
                            _train, _target = balance_out(balance_type, train_scaled, dum_train_y, seed, BALANCE_OPTS)
                        except Exception as e:
                            st.warning(str(e))
                    if balance_type != "DEFAULT":
                        st.subheader("Train Data (BALANCED)")
                        
                        st.dataframe(_train.head(50))
                        st.write(_train.shape)
                        st.markdown(download_csv(_train, "cpt_train_balanced.csv", info="DOWNLOAD BALANCED TRAIN FILE"), unsafe_allow_html=True)
            
                else:
                    _train, _target = train_scaled.copy(), dum_train_y.copy()
            else: #unique_class > 2:
                st.write(f"{unique_class} Classes Observed....")
 
            

            st.header('TRAINING/TESTING SECTION')

            model = st.sidebar.selectbox('Select Algorithm: ', MODELS)
            
            
            #algorithm selection and hyperparameter tuning
            params = model_parameter(model)
            model_ = build_model(model, params, seed)


            _, val_, test_, test_resp = initialize_model(model=model_, Xtrain_file=_train, ytrain_file=_target["target"], \
                                                        test_file=test_scaled, test_dataframe=test_id, target_var_=target_col[0], seed=seed)

            if test_resp is not None:
                # st.write("Train Accuracy (on train data: ", sklearn.metrics.accuracy_score(train_[0], train_[1]))
                st.write("VALIDATION Accuracy (on train data): ", np.round(sklearn.metrics.accuracy_score(val_[0], val_[1])*100, 1), '(%)')
                st.write("TEST Accuracy (on train data): ", np.round(sklearn.metrics.accuracy_score(test_[0], test_[1])*100, 1), '(%)')

                st.write("TEST F1 SCORE (on train data): ", np.round(sklearn.metrics.f1_score(test_[0], test_[1])*100, 1), '(%)')

                st.write(test_resp.head(1000))
                st.write(test_resp.shape)
                st.write("")
                st.markdown(download_csv(test_resp, "MLCPT_TEST_PRED.csv", info="DOWNLOAD TEST PREDICTION FILE"), unsafe_allow_html=True)
                st.write("MODEL ESTABLISHED. YAY!")
                target_count = len(set(test_resp['target']))
                st.write(f'{target_count} classes observed. {test_resp.shape}')

                with st.spinner('Prediction Data'):
                    target_col = 'target'
                    target_cp, ax = plt.subplots()
                    sns.countplot(data=test_resp, x=target_col)
                    st.pyplot(target_cp)
                plot_target()

                # celebrate = on_completion() #version does not support snow
                # celebrate()
                st.balloons()
                

                del train_scaled, test_scaled, train, test, df
            else:
                st.write("YOUR MODEL FAILED TO COMPLETE")
            

        elif train_df:
            st.write("YOU NEED TEST DATASET TOO")
        elif test_df:
            st.write("YOU NEED TRAIN DATASET AS WELL")
        else:
            st.write("ABEG UPLOAD TRAIN AND TEST DATASET")
    else:
        st.write('INVALID ARGUMENT! ')

    st.markdown("<h5 style='text-align: center'>Made with <span style='color:red'>&hearts;</span> By <a href='https://www.twitter.com/__oemmanuel__'>Emmanuel</a> </h5>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
