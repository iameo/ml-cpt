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
     remove_mono_unique, download_csv, model_parameter, build_model, initialize_model, download_csv)



pd.options.mode.chained_assignment = None

plt.rcParams.update({'figure.max_open_warning':0})
sns.set_theme(style="darkgrid")


MODELS = ['CATBOOST', 'KNN', 'RANDOMFOREST', 'XGBOOST']
SCALER = ["STANDARDSCALER", "MIN-MAX SCALER"]




def main():
    st.title('MACHINE LEARNING FOR YOU..')

    welcome_text = st.markdown(get_content("README.md"))
    options = ['WELCOME', 'EXPLORE']
    option = st.sidebar.selectbox('Select option: ', options)

    if option == options[0]:
        pass
    elif option == options[1]:
        
        #ensuring producibility
        seed = st.sidebar.slider('SEED', 1, 50, step=1)

        np.random.seed(seed=seed)
        np.random.RandomState(seed=seed)
        
        welcome_text.empty()
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
            df, mem_reduced = reduce_mem_usage(df)
            st.write("MEMORY SAVED: ", mem_reduced,"MB")
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
            test_data.loc[test_data["marker"] == "test", target_col[0]] = "N/A" #
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

            #conserve memory
            df = None

            if full_train is not None and full_test is not None:
                new_df = pd.concat([full_train, full_test], axis=0) #use padded data
            else:
                new_df = pre_miss_df #use this since missing data wasn't present
            st.dataframe(new_df.head(50))
            st.write("SHAPE: ", new_df.shape)
            if new_df.shape[1] > 50:
                st.write("ABSOLUTE CORRELATION WITH TARGET VARIABLE")
                st.write(new_df[new_df["marker"] == "train"].corr()[target_col[0]].sort_values(by=target_col[0], ascending=False).T)
                st.write("[correlation is not causation]")
            else:
                heatmap_fig, ax=plt.subplots()
                sns.heatmap(new_df[new_df["marker"] == "train"].corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
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


            not_dummy = [target_col[0], "target", "marker", "claim", "prediction", "response"]
            exclude_cols = [col for col in new_df.columns if col not in not_dummy]
            exclude_cols = list(set(exclude_cols).intersection(list(new_df.select_dtypes(include='object').columns)))
            dum_df = pd.get_dummies(new_df, columns=exclude_cols, drop_first=True)
            st.dataframe(dum_df.head(100))
            st.write(dum_df.shape)
            st.write("CATEGORICAL FEATURES ENCODED")

            new_df = None

            dum_train = dum_df[dum_df["marker"] == "train"].drop([target_col[0], "marker"], axis=1)
            dum_train_y = dum_df[dum_df["marker"] == "train"][target_col[0]].astype(int)
            dum_test = dum_df[dum_df["marker"] == "test"].drop([target_col[0], "marker"], axis=1)
        

            scaler_option = st.selectbox("SCALE DATA USING: ", SCALER)
            if scaler_option == SCALER[0]:
                from sklearn.preprocessing import StandardScaler
                ss = StandardScaler()
                Xtrain = pd.DataFrame(ss.fit_transform(dum_train), columns=dum_train.columns)
                test = pd.DataFrame(ss.transform(dum_test), columns=dum_test.columns)
            elif scaler_option == SCALER[1]:
                from sklearn.preprocessing import MinMaxScaler
                mm = MinMaxScaler()
                Xtrain = pd.DataFrame(mm.fit_transform(dum_train), columns=dum_train.columns)
                test = pd.DataFrame(mm.transform(dum_test), columns=dum_test.columns)
            else:
                st.write("NO SCALER METHOD SELECTED")

            st.subheader("Train Data")
            
            st.dataframe(Xtrain.head(1000))
            st.write(Xtrain.shape)
            st.markdown(download_csv(Xtrain, "cpt_train.csv", info="DOWNLOAD TRAIN FILE"), unsafe_allow_html=True)
            
            st.subheader("Test Data")
            
            st.dataframe(test.head(1000))
            st.write(test.shape)
            st.markdown(download_csv(test, "cpt_test.csv", info="DOWNLOAD TEST FILE"), unsafe_allow_html=True)

            st.header('TRAINING/TESTING SECTION')

            model = st.sidebar.selectbox('Select Algorithm: ', MODELS)
            

            params = model_parameter(model)
            model_ = build_model(model, params, seed)



            test_resp, y_test_, y_test = initialize_model(model=model_, Xtrain_file=Xtrain, ytrain_file=dum_train_y, test_file=test, test_dataframe=test_id, target_var_=target_col[0], seed=seed)

            if test_resp is not None and y_test_ is not None and y_test is not None:
                st.write("TEST ACCURACY: ", sklearn.metrics.accuracy_score(y_test_, y_test))
                st.write("TEST F1 SCORE: ", sklearn.metrics.f1_score(y_test_, y_test))
                # pred_dict = {"ID": np.array(test_id.values), target_col[0]: model_.predict(test)}
                # test_pred = pd.DataFrame.from_dict(pred_dict)
                st.write("")
                st.write(test_resp.head(1000))
                st.write(test_resp.shape)
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