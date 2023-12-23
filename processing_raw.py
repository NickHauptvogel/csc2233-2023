import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
# filepath should point to the directory where all the csv or data files located
for file_number in range(0, 6):
    if file_number == 1:
        filepath = "data_2022_Q4"
        first_csv = "/2022-10-01.csv"
        year = "2022"
    elif file_number == 2:
        filepath = "data_2023_Q1"
        first_csv = "/2023-01-01.csv"
        year = "2023"
    elif file_number == 3:
        filepath = "data_2023_Q2"
        first_csv = "/2023-04-01.csv"
        year = "2023"
    elif file_number == 4:
        filepath = "data_2023_Q3_P1"
        first_csv = "/2023-07-01.csv"
        year = "2023"
    elif file_number == 5:
        filepath = "data_2023_Q3_P2"
        first_csv = "/2023-09-01.csv"
        year = "2023"

    # list all the valid file names
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    files = [f for f in files if "csv" in f and f.startswith(year)]
    len(files)
    files.sort()

    # determine all the columns that have to be dropped by default
    def drop_column_names():
        data = pd.read_csv(filepath + first_csv)
        drop_columns = []   
        return drop_columns        
    # columns that have to be dropped
    drop_names = drop_column_names()
    # put all the dataframes into a list 
    master_df_list = []
    # iterate over each file and drop the unwanted columns
    for file_name in files:
        data = pd.read_csv(filepath + "/" +file_name)
        data.drop(drop_names, axis = 1, inplace=True)
        master_df_list.append(data.copy())
        print("{} added to master_list".format(file_name))
    
    # generate a master data frame to have all the data
    master_df = pd.concat(master_df_list)
    master_df.shape
    
    # determine the count of each model in entire dataset
    models_count = master_df["model"]
    models_count = models_count.value_counts().to_frame()
    models_count.columns = ["count"]
    models_count['name'] = models_count.index
    models_count.reset_index(drop=True)
    models_count.shape
    
    # determine the failure_count of each model in entire dataset
    failure_counts = master_df[master_df["failure"]==1].groupby(["model"]).agg({'model':'count'})
    failure_counts.columns = ["failed_count"]
    failure_counts['name'] = failure_counts.index
    failure_counts = failure_counts.reset_index(drop=True)
    failure_counts.shape
    
    # determine the no_failure_count of each model in entire dataset
    no_failure_counts = master_df[master_df["failure"]==0].groupby(["model"]).agg({'model':'count'})
    no_failure_counts.columns = ["no_failed_count"]
    no_failure_counts['name'] = no_failure_counts.index
    no_failure_counts = no_failure_counts.reset_index(drop=True)
    no_failure_counts.shape
    
    # combine no_failure_counts and failure_counts into one df
    merged_level_1 = pd.merge(no_failure_counts, failure_counts, left_on='name', right_on='name', how="left")
    merged_level_1
    
    # combine models_count and failure_counts into one df
    merged_level_2 = pd.merge(models_count, merged_level_1, left_on='name', right_on='name', how="left")
    merged_level_2 = merged_level_2.fillna(0)
    merged_level_2["failed_count"] = merged_level_2["failed_count"].astype('int32')
    merged_level_2 = merged_level_2.sort_values(by='failed_count', ascending=False)
    merged_level_2
    base_stats = filepath+"_"+"base_stats.csv"
    merged_level_2.to_csv(base_stats, index = False)
    # Analysis of Model 1 EXPERIMENTING -

    # set the model name here 
    # model_name = "ST4000DM000" 
    # model_name = "ST8000NM0055" 
    model_name = "TOSHIBA MG07ACA14TA" 

    
    model_1 = master_df[master_df["model"] == model_name]

    # drop all columns that are only NaN
    model_1 = model_1.dropna(axis=1, how='all') # 67 to 27 column reduction
    model_1
    cols = model_1.select_dtypes([np.number]).columns
    std = model_1[cols].std()
    cols_to_drop = std[std==0].index

    # drops all rows with even one NaN value
    print(model_1.shape)
    model_1 = model_1.dropna()
    print(model_1.shape)
    # find all columns that are filled with the same value
    columns_with_same_values = model_1.columns[model_1.nunique() <= 1].tolist()
    # do not remove the columns 'model' and 'capacity_bytes'
    columns_with_same_values.remove('model')
    columns_with_same_values.remove('capacity_bytes')
    columns_with_same_values
    print(model_1.shape)
    model_1 = model_1.drop(columns_with_same_values, axis=1)
    print(model_1.shape)
    model_1 = model_1.sort_values(['serial_number', 'date'], ascending=[True, True])
    # FINAL CLEANING CODE
    # data cleaning operations
    def data_cleaning(model_name):
        model_1 = master_df[master_df["model"] == model_name]
        
        # drop all columns that are only NaN
        model_1 = model_1.dropna(axis=1, how='all')
        print("initial column drop done")
        
        # drops all rows with even one NaN value
        model_1 = model_1.dropna()
        print("initial row drop done")
        
        # find all columns that are filled with the same value and drop them
        columns_with_same_values = model_1.columns[model_1.nunique() <= 1].tolist()
        
        # do not remove the columns 'model' and 'capacity_bytes'
        columns_with_same_values.remove('model')
        columns_with_same_values.remove('capacity_bytes')
        model_1 = model_1.drop(columns_with_same_values, axis=1)
        print("repeating cloumn value drop done")
        
        # sort the dataframe based on serial_number and date
        model_1 = model_1.sort_values(['serial_number', 'date'], ascending=[True, True])
        print("sort done")
        return model_1
    # cleaned data

    destination_path = model_name + "_RAW_Dec11/"

    cleaned_data = data_cleaning(model_name)
    # export data
    export_filename = filepath+"_"+model_name+".csv"
    cleaned_data.to_csv(os.path.join(destination_path,export_filename), index = False)
    print("export complete")
