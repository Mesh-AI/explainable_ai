#import necessary libraries

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
base_dir = Path('/Users/madalinaciolan/dev/explainable_ai')

#load the energy demand data(load)
load_path = base_dir / 'data' / 'Load' / 'load_history.csv'
load_df = pd.read_csv(load_path)

#load the temperature data
temp_path = base_dir / 'data' / 'Load' / 'temperature_history.csv'
temp_df = pd.read_csv(temp_path)

#fix the index for both dataframes
load_df.index = range(1, len(load_df) + 1)
print(load_df.head())

temp_df.index = range(1, len(temp_df) + 1)
print(temp_df.head())

#check the info and missing values for both dataframes
print(load_df.info())
print(load_df.isna().sum())

print(temp_df.info())
print(temp_df.isna().sum())

# ---- Clean the load dataframe ----   
load_df_clean = load_df.copy()
cols = [f"h{i}" for i in range(1, 25)]
for col in cols:
    load_df_clean[col] = load_df_clean[col].str.replace(",", "").astype(float)

print(load_df_clean.dtypes)
print(load_df_clean.isna().sum())

# --- Fill NaNs in load_df_clean using interpolation before/after ---
# 1) Horizontal interpolation within each day across h1..h24
#    (fills gaps inside a day's 24 hours using surrounding hours)
load_df_clean[cols] = load_df_clean[cols].interpolate(method='linear', axis=1, limit_direction='both')

# 2) Vertical interpolation across days for each hour h1..h24
#    (fills gaps across days using surrounding days for each hour)
load_df_clean[cols] = load_df_clean[cols].interpolate(method='linear',  axis=0, limit_direction='both')

# check if all NaNs in the load data frame have been filled
print("\nRemaining NaNs per column after interpolation: ", load_df_clean[cols].isna().sum())


# ---- Clean the temperature dataframe ----  
print(load_df_clean.dtypes) #no need to change types, they are already integers and floats
print(temp_df.isna().sum()) #NaNs only in temperature colums h7 -> h24
temp_df_clean = temp_df.copy()

cols_temp = [f"h{i}" for i in range(7, 25)]
for col in cols_temp:
    temp_df_clean[col] = temp_df_clean[col].interpolate(method='linear', axis=0, limit_direction='both')    

#check if all NaNs in the temperature data frame have been filled
print("\nRemaining NaNs per column after interpolation: ", temp_df_clean.isna().sum()) 

print(load_df_clean.head())


# ----Reshaping the load dataframe from wide to long format ----

# Melt the dataframe to have a long format with columns: year, month, day, hour, load
hours_cols = [f"h{i}" for i in range(1, 25)]
load_long = load_df_clean.melt(
    id_vars=["zone_id", "year", "month", "day"],
    value_vars=hours_cols,
    var_name="hour",
    value_name="load",
)

# Convert hour column to numeric hour
load_long['hour'] = load_long['hour'].str.extract('h(\d+)').astype(int)

# Create a datetime column
load_long['date_time'] = pd.to_datetime(load_long[['year', 'month', 'day']]) + pd.to_timedelta(load_long['hour'] - 1, unit='h')

#Keep only relevant columns and sort by date
load_long = load_long[['zone_id','date_time', 'load']].sort_values(['zone_id', 'date_time']).reset_index(drop=True)
print(load_long.head())

print(temp_df_clean.head())

# ----Reshaping the temperature dataframe from wide to long format ----
# Melt the dataframe to have a long format with columns: year, month, day, hour, temperature
temp_hour_cols = [f"h{i}" for i in range(1, 25)]
temp_long = temp_df_clean.melt(
    id_vars=["station_id", "year", "month", "day"],
    value_vars=temp_hour_cols,
    var_name="hour",
    value_name="temperature",
)

# Convert hour column to numeric hour
temp_long['hour'] = temp_long['hour'].str.extract('h(\d+)').astype(int)

# Create a datetime column
temp_long['date_time'] = pd.to_datetime(temp_long[['year', 'month', 'day']]) + pd.to_timedelta(temp_long['hour'] - 1, unit='h')

#Keep only relevant columns and sort by date
temp_long = temp_long[['station_id','date_time', 'temperature']].sort_values(by='date_time').reset_index(drop=True)
print(temp_long.head())

# ---- Merge load and temperature dataframes on date_time ----
# Assuming zone_id and station_id correspond to the same geographical area
merged_df = pd.merge(load_long, temp_long, left_on=['zone_id', 'date_time'], right_on=['station_id', 'date_time'], how='left')
merged_df = merged_df.drop(columns=['station_id'])  # Drop redundant station_id column
print(merged_df.head(24))
