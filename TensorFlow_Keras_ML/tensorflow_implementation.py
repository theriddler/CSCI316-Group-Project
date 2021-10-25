# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# define folder path
path = "../fonts/"

# list each file in folder
files = os.listdir(path)

# create list of DataFrames
df_list = [pd.read_csv(path+file) for file in files]

# concatenate DFs together
data = pd.concat(df_list)


n_rows = data.shape[0]
n_columns = data.shape[1]
print(data)