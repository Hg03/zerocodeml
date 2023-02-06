import pandas as pd
import streamlit as st
import ydata_profiling as pp
from streamlit_pandas_profiling import st_profile_report as spp
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid
from pycaret.classification import *
from pycaret.regression import *
import requests
from PIL import Image

img = Image.open('assets/deeplearning.png')
st.set_page_config(layout='wide',page_title='zerocodeml',page_icon = img)

@st.cache
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def visualize_data(df):
    st.subheader('Sample Data')
    AgGrid(df.head(10))
    report = df.profile_report()
    spp(report)

def pre_regression(df,num_feat,cat_feat,independent_feat,ignore_feat,target):
    
    pre_steps = {'Impute The Missing Values (Numeric)':['mean','median','zero'],'Impute The Missing Values (Categorical)':['mode','not available']}
    to_do = {'Impute The Missing Values (Numeric)':None,'Impute The Missing Values (Categorical)':None}
    steps = st.multiselect('To clean the data and get the accurate model',pre_steps.keys())
    for step in range(len(steps)):
        to_do[steps[step]] = st.selectbox(steps[step],pre_steps[steps[step]])
    encode,remove_outliers,normalize = None,None,None
    one, two, three = st.columns(3)
    with one:
        encode = st.checkbox('Do you want to Encode the categorical feature ?')
    with two:
        remove_outliers = st.checkbox('Do you want to Remove the outliers ?')
    with three:
        normalize = st.checkbox('Do you want to Normalize the feature ?')
    if encode:
        ordinal_feat = st.multiselect('Please specify the ordinal feature',cat_feat)
        ordinal_features = {}
        for i in ordinal_feat:
            ordinal_features[i] = list(df[i].unique())

    # All selected 6c6
    if (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat):
        reg = setup(data=df, target=target, numeric_features=num_feat, categorical_features=cat_feat, ignore_features=ignore_feat, ordinal_features=ordinal_features, numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True,normalize=True)
    
    # Nothing selected 6c0
    elif not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat):
        reg = setup(data = df, target=target,numeric_features=num_feat,categorical_features=cat_feat)  
    
    # One is selected only 6c1
    elif (to_do['Impute The Missing Values (Numeric)'] and not (to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'])
    elif (to_do['Impute The Missing Values (Categorical)'] and not(to_do['Impute The Missing Values (Numeric)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (encode and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features)
    elif (remove_outliers and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True)
    elif (normalize and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True)
    elif (ignore_feat and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ignore_feat = ignore_feat)

    # Two are selected only 6c2
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)']) and not (encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif ((to_do['Impute The Missing Values (Numeric)'] and encode) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers) and not (to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categoric_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif((encode and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target, numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,normalize=True)
    elif((encode and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True)
    elif((encode and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif((normalize and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,ignore_features=ignore_feat)
    elif((normalize and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,remove_outliers=True)
    elif((ignore_feat and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True,ignore_features=ignore_feat)
    

    # Three are selected only 6c3
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode) and not (normalize and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize) and not (encode and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and not (encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and not (encode and normalize and remove_outliers)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and encode) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and ignore_feat and encode) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and remove_outliers) and not (to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True,normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat,remove_outliers=True)
    elif ((encode and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True,normalize=True)
    elif ((encode and remove_outliers and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features,ignore_features=ignore_feat,remove_outliers=True)
    elif ((ignore_feat and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, ignore_features=ignore_feat,remove_outliers=True, normalize=True)
    elif ((encode and ignore_feat and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat,normalize=True)
    

    # Four are selected only 6c4
    elif (not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)']) and (encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (not (to_do['Impute The Missing Values (Numeric)'] and encode) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers) and (to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and normalize) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and ignore_feat) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ignore_features=ignore_feat)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and encode) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categoric_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and normalize) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif(not (encode and normalize) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target, numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,normalize=True)
    elif(not (encode and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True)
    elif(not (encode and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif(not (normalize and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,ignore_features=ignore_feat)
    elif(not(normalize and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,remove_outliers=True)
    elif(not (ignore_feat and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True,ignore_features=ignore_feat)

    # 5 are selected only 6c5
    elif (not to_do['Impute The Missing Values (Numeric)'] and (to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'])
    elif (not to_do['Impute The Missing Values (Categorical)'] and (to_do['Impute The Missing Values (Numeric)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (not encode and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features)
    elif (not remove_outliers and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True)
    elif (not normalize and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True)
    elif (not ignore_feat and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ignore_feat = ignore_feat)




def pre_classification(df,num_feat,cat_feat,independent_feat,ignore_feat,target):
    pre_steps = {'Impute The Missing Values (Numeric)':['mean','median','zero'],'Impute The Missing Values (Categorical)':['mode','not available']}
    to_do = {'Impute The Missing Values (Numeric)':None,'Impute The Missing Values (Categorical)':None}
    steps = st.multiselect('To clean the data and get the accurate model',pre_steps.keys())
    for step in range(len(steps)):
        to_do[steps[step]] = st.selectbox(steps[step],pre_steps[steps[step]])
    encode,remove_outliers,normalize = None,None,None
    one, two= st.columns(2)
    with one:
        encode = st.checkbox('Do you want to Encode the categorical feature ?')
    with two:
        remove_outliers = st.checkbox('Do you want to Remove the outliers ?')
    if encode:
        ordinal_feat = st.multiselect('Please specify the ordinal feature',cat_feat)
        ordinal_features = {}
        for i in ordinal_feat:
            ordinal_features[i] = list(df[i].unique())

    # All selected 6c6
    if (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and ignore_feat):
        reg = setup(data=df, target=target, numeric_features=num_feat, categorical_features=cat_feat, ignore_features=ignore_feat, ordinal_features=ordinal_features, numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    
    # Nothing selected 6c0
    elif not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat):
        reg = setup(data = df, target=target,numeric_features=num_feat,categorical_features=cat_feat)  
    
    # One is selected only 6c1
    elif (to_do['Impute The Missing Values (Numeric)'] and not (to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'])
    elif (to_do['Impute The Missing Values (Categorical)'] and not(to_do['Impute The Missing Values (Numeric)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (encode and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features)
    elif (remove_outliers and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True)
    elif (normalize and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True)
    elif (ignore_feat and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ignore_feat = ignore_feat)

    # Two are selected only 6c2
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)']) and not (encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif ((to_do['Impute The Missing Values (Numeric)'] and encode) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers) and not (to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categoric_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif((encode and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target, numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,normalize=True)
    elif((encode and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True)
    elif((encode and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif((normalize and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,ignore_features=ignore_feat)
    elif((normalize and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,remove_outliers=True)
    elif((ignore_feat and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True,ignore_features=ignore_feat)
    

    # Three are selected only 6c3
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode) and not (normalize and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize) and not (encode and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and not (encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and not (encode and normalize and remove_outliers)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and encode) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,normalize=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and ignore_feat and encode) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and remove_outliers) and not (to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,remove_outliers=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat) and not (to_do['Impute The Missing Values (Numeric)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True,normalize=True)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat and normalize) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True,ignore_features=ignore_feat)
    elif ((to_do['Impute The Missing Values (Categorical)'] and ignore_feat and remove_outliers) and not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat,remove_outliers=True)
    elif ((encode and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True,normalize=True)
    elif ((encode and remove_outliers and ignore_feat) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features,ignore_features=ignore_feat,remove_outliers=True)
    elif ((ignore_feat and remove_outliers and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, ignore_features=ignore_feat,remove_outliers=True, normalize=True)
    elif ((encode and ignore_feat and normalize) and not (to_do['Impute The Missing Values (Categorical)'] and to_do['Impute The Missing Values (Numeric)'] and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat,normalize=True)
    

    # Four are selected only 6c4
    elif (not (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)']) and (encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (not (to_do['Impute The Missing Values (Numeric)'] and encode) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ordinal_features=ordinal_features)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and remove_outliers) and (to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],remove_outliers=True)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and normalize) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],normalize=True)
    elif (not (to_do['Impute The Missing Values (Numeric)'] and ignore_feat) and (to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'],ignore_features=ignore_feat)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and encode) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ordinal_features=ordinal_features)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categoric_imputation=to_do['Impute The Missing Values (Categorical)'],remove_outliers=True)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and normalize) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],normalize=True)
    elif (not (to_do['Impute The Missing Values (Categorical)'] and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and remove_outliers and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'],ignore_features=ignore_feat)
    elif(not (encode and normalize) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and ignore_feat)):
        reg = setup(data=df,target=target, numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,normalize=True)
    elif(not (encode and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat,ordinal_features=ordinal_features,remove_outliers=True)
    elif(not (encode and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat, categorical_features=cat_feat, ordinal_features=ordinal_features,ignore_features=ignore_feat)
    elif(not (normalize and ignore_feat) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,ignore_features=ignore_feat)
    elif(not(normalize and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True,remove_outliers=True)
    elif(not (ignore_feat and remove_outliers) and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True,ignore_features=ignore_feat)

    # 5 are selected only 6c5
    elif (not to_do['Impute The Missing Values (Numeric)'] and (to_do['Impute The Missing Values (Categorical)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,numeric_imputation=to_do['Impute The Missing Values (Numeric)'])
    elif (not to_do['Impute The Missing Values (Categorical)'] and (to_do['Impute The Missing Values (Numeric)'] and encode and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,categorical_imputation=to_do['Impute The Missing Values (Categorical)'])
    elif (not encode and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ordinal_features=ordinal_features)
    elif (not remove_outliers and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and encode and normalize and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,remove_outliers=True)
    elif (not normalize and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and encode and ignore_feat)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,normalize=True)
    elif (not ignore_feat and (to_do['Impute The Missing Values (Numeric)'] and to_do['Impute The Missing Values (Categorical)'] and remove_outliers and normalize and encode)):
        reg = setup(data=df,target=target,numeric_features=num_feat,categorical_features=cat_feat,ignore_feat = ignore_feat)

def preprocess_data(df):
    target = st.selectbox('Please select the target variable',list(df.columns))
    if df[target].nunique() > 6:
        st.success('Looks like it is a regression problem')
    else:
        st.success('Looks like it is a classification problem')
    independent_feat = [col for col in df.columns if col != target]
    ignore_feat = st.multiselect('Some columns you want to ignore, so i can do it also',independent_feat)
    independent_feat = [col for col in independent_feat if col not in ignore_feat]
    num_feat = [col for col in independent_feat if df[col].dtype != 'O']
    cat_feat = [col for col in independent_feat if col not in num_feat]
    col3, col4 = st.columns(2)
    with col3:
        with st.expander(f'Number of numerical features present are {len(num_feat)}'):
            AgGrid(df[num_feat].head(6))
    
    with col4:
        with st.expander(f'Number of categorical features present are {len(cat_feat)}'):
            AgGrid(df[cat_feat].head(6))
    

    col1, col2 = st.columns(2)
    preprocess_url = 'https://assets2.lottiefiles.com/packages/lf20_iitdh6nn.json'
    preprocess_icon = load_lottieurl(preprocess_url)
    col1.markdown("<h1 style='text-align: center; '>Now let's proceed with \nPreprocessing<br> Steps</h1>", unsafe_allow_html=True)
    with col2:
        st_lottie(preprocess_icon,width=240)
    if df[target].nunique() > 6:
        pre_regression(df,num_feat,cat_feat,independent_feat,ignore_feat,target)
    else:
        pre_classification(df,num_feat,cat_feat,independent_feat,ignore_feat,target) 
    


col1,col2 = st.columns(2)
url = 'https://assets3.lottiefiles.com/packages/lf20_8CeqKMzpWz.json'
url_side = 'https://assets6.lottiefiles.com/packages/lf20_xafe7wbh.json'
url_to_json = load_lottieurl(url)
url_side_to_json = load_lottieurl(url_side)
col1.markdown("# Zero Code ML\n ### Don't worry about coding, I'll take care of them.\n### Just do as I say 🪄")
with col2:
    st_lottie(url_to_json,width=250)

st.info('👈 Upload your dataset ')

with st.sidebar:
    st_lottie(url_side_to_json)
st.sidebar.title('Upload your dataset (CSV or Excel file)')

data = st.sidebar.file_uploader('Your data',type=['csv'])
if data is not None:
    name_of_file = data.name 
    extension = name_of_file[name_of_file.rfind('.')+1:]
    if extension not in ['csv']:
        st.sidebar.warning('Please upload valid dataset')
    else:
        df = pd.read_csv(data)

        select_ = st.sidebar.radio('Select what you need ?',['Visualize','Preprocess','Download'])
        if select_ == 'Visualize':
            visualize_data(df)
        if select_ == 'Preprocess':
            preprocess_data(df)



            

