import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# First run
# sources = ["data_2022_Q4_ST4000DM000.csv", "data_2023_Q1_ST4000DM000.csv", "data_2023_Q2_ST4000DM000.csv", "data_2023_Q3_P1_ST4000DM000.csv", "data_2023_Q3_P2_ST4000DM000.csv"]
# sources = ["ST8000NM0055_RAW_Nov26/data_2022_Q4_ST8000NM0055.csv", "ST8000NM0055_RAW_Nov26/data_2023_Q1_ST8000NM0055.csv",  "ST8000NM0055_RAW_Nov26/data_2023_Q2_ST8000NM0055.csv", "ST8000NM0055_RAW_Nov26/data_2023_Q3_P1_ST8000NM0055.csv", "ST8000NM0055_RAW_Nov26/data_2023_Q3_P2_ST8000NM0055.csv"]
# sources = ["ST12000NM0008_RAW_Nov26/data_2022_Q4_ST12000NM0008.csv", "ST12000NM0008_RAW_Nov26/data_2023_Q1_ST12000NM0008.csv", "ST12000NM0008_RAW_Nov26/data_2023_Q2_ST12000NM0008.csv","ST12000NM0008_RAW_Nov26/data_2023_Q3_P1_ST12000NM0008.csv","ST12000NM0008_RAW_Nov26/data_2023_Q3_P2_ST12000NM0008.csv"]

sources = ["TOSHIBA MG07ACA14TA_RAW_Dec11/data_2022_Q4_TOSHIBA MG07ACA14TA.csv", "TOSHIBA MG07ACA14TA_RAW_Dec11/data_2023_Q1_TOSHIBA MG07ACA14TA.csv", "TOSHIBA MG07ACA14TA_RAW_Dec11/data_2023_Q2_TOSHIBA MG07ACA14TA.csv", "TOSHIBA MG07ACA14TA_RAW_Dec11/data_2023_Q3_P1_TOSHIBA MG07ACA14TA.csv", "TOSHIBA MG07ACA14TA_RAW_Dec11/data_2023_Q3_P2_TOSHIBA MG07ACA14TA.csv"]

file_names = [f for f in sources if isfile(f)]
import csv
total = 0

for csv_file in file_names:
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Use len() to get the number of rows in the CSV file
        row_count = len(list(csv_reader))
        total += row_count
        print(f"The number of rows in {csv_file} is: {row_count}")
print(f"The Total is {total}")

# Initialize an empty DataFrame as the target DataFrame
target_df = pd.read_csv(file_names[0])
# Iterate through each source file
for source in file_names[1:]:
    # Read the source CSV file into a DataFrame
    source_df = pd.read_csv(source)

    # Identify common columns
    common_columns = set(target_df.columns) & set(source_df.columns)
    print(f"These are the common columns {common_columns}")

    # Select only the common columns from the source DataFrame
    source_df_common = source_df[list(common_columns)]

    # Merge DataFrames based on common columns
    target_df = pd.merge(target_df, source_df_common, how='outer', left_on=list(common_columns), right_on=list(common_columns))

# Select only the common columns from the source DataFrame
desired_columns_order = [
    'date', 'serial_number', 'model', 'capacity_bytes', 'failure',
    'smart_1_normalized', 'smart_1_raw', 'smart_2_normalized', 'smart_2_raw',
    'smart_3_normalized', 'smart_3_raw', 'smart_4_normalized', 'smart_4_raw',
    'smart_5_normalized', 'smart_5_raw', 'smart_7_normalized', 'smart_7_raw',
    'smart_8_normalized', 'smart_8_raw', 'smart_9_normalized', 'smart_9_raw',
    'smart_10_normalized', 'smart_10_raw', 'smart_11_normalized', 'smart_11_raw',
    'smart_12_normalized', 'smart_12_raw', 'smart_13_normalized', 'smart_13_raw',
    'smart_15_normalized', 'smart_15_raw', 'smart_16_normalized', 'smart_16_raw',
    'smart_17_normalized', 'smart_17_raw', 'smart_18_normalized', 'smart_18_raw',
    'smart_22_normalized', 'smart_22_raw', 'smart_23_normalized', 'smart_23_raw',
    'smart_24_normalized', 'smart_24_raw', 'smart_160_normalized', 'smart_160_raw',
    'smart_161_normalized', 'smart_161_raw', 'smart_163_normalized', 'smart_163_raw',
    'smart_164_normalized', 'smart_164_raw', 'smart_165_normalized', 'smart_165_raw',
    'smart_166_normalized', 'smart_166_raw', 'smart_167_normalized', 'smart_167_raw',
    'smart_168_normalized', 'smart_168_raw', 'smart_169_normalized', 'smart_169_raw',
    'smart_170_normalized', 'smart_170_raw', 'smart_171_normalized', 'smart_171_raw',
    'smart_172_normalized', 'smart_172_raw', 'smart_173_normalized', 'smart_173_raw',
    'smart_174_normalized', 'smart_174_raw', 'smart_175_normalized', 'smart_175_raw',
    'smart_176_normalized', 'smart_176_raw', 'smart_177_normalized', 'smart_177_raw',
    'smart_178_normalized', 'smart_178_raw', 'smart_179_normalized', 'smart_179_raw',
    'smart_180_normalized', 'smart_180_raw', 'smart_181_normalized', 'smart_181_raw',
    'smart_182_normalized', 'smart_182_raw', 'smart_183_normalized', 'smart_183_raw',
    'smart_184_normalized', 'smart_184_raw', 'smart_187_normalized', 'smart_187_raw',
    'smart_188_normalized', 'smart_188_raw', 'smart_189_normalized', 'smart_189_raw',
    'smart_190_normalized', 'smart_190_raw', 'smart_191_normalized', 'smart_191_raw',
    'smart_192_normalized', 'smart_192_raw', 'smart_193_normalized', 'smart_193_raw',
    'smart_194_normalized', 'smart_194_raw', 'smart_195_normalized', 'smart_195_raw',
    'smart_196_normalized', 'smart_196_raw', 'smart_197_normalized', 'smart_197_raw',
    'smart_198_normalized', 'smart_198_raw', 'smart_199_normalized', 'smart_199_raw',
    'smart_200_normalized', 'smart_200_raw', 'smart_201_normalized', 'smart_201_raw',
    'smart_202_normalized', 'smart_202_raw', 'smart_206_normalized', 'smart_206_raw',
    'smart_210_normalized', 'smart_210_raw', 'smart_218_normalized', 'smart_218_raw',
    'smart_220_normalized', 'smart_220_raw', 'smart_222_normalized', 'smart_222_raw',
    'smart_223_normalized', 'smart_223_raw', 'smart_224_normalized', 'smart_224_raw',
    'smart_225_normalized', 'smart_225_raw', 'smart_226_normalized', 'smart_226_raw',
    'smart_230_normalized', 'smart_230_raw', 'smart_231_normalized', 'smart_231_raw',
    'smart_232_normalized', 'smart_232_raw', 'smart_233_normalized', 'smart_233_raw',
    'smart_234_normalized', 'smart_234_raw', 'smart_235_normalized', 'smart_235_raw',
    'smart_240_normalized', 'smart_240_raw', 'smart_241_normalized', 'smart_241_raw',
    'smart_242_normalized', 'smart_242_raw', 'smart_244_normalized', 'smart_244_raw',
    'smart_245_normalized', 'smart_245_raw', 'smart_246_normalized', 'smart_246_raw',
    'smart_247_normalized', 'smart_247_raw', 'smart_248_normalized', 'smart_248_raw',
    'smart_250_normalized', 'smart_250_raw', 'smart_251_normalized', 'smart_251_raw',
    'smart_252_normalized', 'smart_252_raw', 'smart_254_normalized', 'smart_254_raw',
    'smart_255_normalized', 'smart_255_raw'
]

tiny_col_order = []

for c in desired_columns_order:
    if c in target_df.columns:
        tiny_col_order.append(c)

target_df = target_df[tiny_col_order]

# Group by serial_number
grouped = target_df.groupby('serial_number')

# Iterate through groups and create CSV files
for serial, group_df in grouped:
    # Construct the file name
    if any(group_df["failure"] == True):
        file_name = f"1_{group_df['model'].iloc[0]}_{serial}.csv"

        # The column contains at least one True value
    else:
        file_name = f"0_{group_df['model'].iloc[0]}_{serial}.csv"

    # Specify the folder path where you want to save the CSV file
    # folder_path = "ST8000NM0055_final_dataset_Nov25/"
    # folder_path = "ST12000NM0008_final_dataset_Nov26/"
    folder_path = "TOSHIBA MG07ACA14TA_final_dataset_Dec11/"
    
    group_df.to_csv(os.path.join(folder_path,file_name))



