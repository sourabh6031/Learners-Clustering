import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st

def preprocess_data(dataframe):
    """Handle missing values, convert categorical to numerical, and clean the dataset"""
    
    ## Filling NaN values in 'job_position' based on the mode of each 'ctc' group
    mode_mapping = dataframe.groupby('ctc')['job_position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    dataframe['job_position'] = dataframe['job_position'].fillna(dataframe['ctc'].map(mode_mapping))

    ## If NaN values remain, checking their percentage and handle them
    nan_ratio = dataframe['job_position'].isna().sum() / len(dataframe)

    if nan_ratio > 0.05:
        dataframe['job_position'].fillna('Others', inplace=True)  # Filling remaining NaNs with 'Others'
    else:
        dataframe.dropna(subset=['job_position'], inplace=True)  # Droping rows where 'job_position' is still NaN

    
    # For Numerical value
    dataframe['ctc_updated_year'] = dataframe.groupby('job_position')['ctc_updated_year'].transform(lambda x: x.fillna(x.mean()))
    dataframe['orgyear'] = dataframe.groupby('job_position')['orgyear'].transform(lambda x: x.fillna(x.mean()))
    dataframe['ctc'] = dataframe.groupby('job_position')['ctc'].transform(lambda x: x.fillna(x.mean()))  
    
    
    return dataframe

def feature_engineering(df):
    """Add new columns based on existing features"""
    current_year  = 2025 # assuming current year
    df['YOE'] = current_year - df['orgyear']
    df['years_for_first_increment'] = df.loc[:,'ctc_updated_year'] - df.loc[:,'orgyear']
    
    # setting correct data type
    df['YOE'] = df.loc[:,'YOE'].astype('int')
    df['years_for_first_increment'] = df.loc[:,'years_for_first_increment'].astype('int')


    # calculating means() for new column
    job_avg_ctc = df.groupby(['job_position'])['ctc'].mean().reset_index(name = 'avg_ctc')                     # calculating the avg by just job_position
    avg_yoe_position = df.groupby(['job_position','YOE'])['ctc'].mean().reset_index(name = 'avg_position_ctc') # calculating the avg by job_position and YOE
    
    # merging to form new column
    df = pd.merge(df,job_avg_ctc,on=['job_position'],how='left')              # merging the avg by just job_position
    df = pd.merge(df,avg_yoe_position,on=['job_position','YOE'],how='left')   # merging the avg by job_position and YOE
    
    # creating 2 new column as a flag for above two column
    df['avg_ctc_flag'] = df.apply(lambda x: 1 if x['ctc'] > x['avg_ctc'] else (2 if x['ctc'] == x['avg_ctc'] else 3), axis=1)
    df['avg_position_ctc_flag'] = df.apply(lambda x: 1 if x['ctc'] > x['avg_position_ctc'] else (2 if x['ctc'] == x['avg_position_ctc'] else 3), axis=1)
    
    # setting the correct data type
    df['avg_ctc_flag'] = df.loc[:,'avg_ctc_flag'].astype('int')
    df['avg_position_ctc_flag'] = df.loc[:,'avg_position_ctc_flag'].astype('int')


    # creating new column "ctc_level"
    ctc_level_code = [
    df['ctc'] < 500000,                     # Below 500,000 → Level 1
    (df['ctc'] >= 500000) & (df['ctc'] <= 1000000),  # Between 500,000 and 1,000,000 → Level 2
    (df['ctc'] > 1000000) & (df['ctc'] <= 1700000)]   # Between 10,00,000 and 17,00,000 → Level 3 ELSE Level 4.
    
    choices = [1, 2, 3]
    df['ctc_level'] = np.select(ctc_level_code, choices, default=4)
    # setting it correct data type
    df['ctc_level'] = df.loc[:,'ctc_level'].astype('int')

    df = df.drop(columns=['job_position'])

    # DUE TO MERGING, SOME DATA HAS BEEN DUPLICATED. SO REMOVING THE DUPLICATED DATA
    df.drop_duplicates(keep='first', inplace=True, ignore_index=False)
   
    
    return df


def feature_scaling(df):
    '''SCALING THE FEATURES'''
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    st.write(round(X_scaled.isna().sum()/len(X_scaled)*100,2))
    return X_scaled


def outlier_treatment(dataframe):
    dataframe = dataframe.dropna()
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.05, metric='euclidean')
    dataframe['lof_score'] = lof.fit_predict(dataframe)
    dataframe['lof_anomaly_score'] = lof.negative_outlier_factor_

    # FILTERING OUT THE ANOMALIES

    dataframe = dataframe[dataframe['lof_score'] == 1]

    dataframe = dataframe.iloc[:,:-2]

    return dataframe 