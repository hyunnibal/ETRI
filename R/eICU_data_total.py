import csv
import numpy as np
import pandas as pd

data = pd.read_csv('./Data/eicu_data_0910.csv',sep = ',',header = 0, index_col = 0)
label = pd.read_csv('./Data/eicu_label_0910.csv',sep = ',',header = 0, index_col = 0)

train_size = round(label.values.shape[0]*0.6)
vali_size = round(label.values.shape[0]*0.2)
test_size = label.values.shape[0] - train_size - vali_size

total = {'X_train': data.values[range(0, train_size)],
         'X_vali': data.values[range(0, vali_size)],
         'X_test': data.values[range(0, test_size)],
         'Y_train': data.values[range(0, train_size)],
         'Y_vali': data.values[range(0, vali_size)],
         'Y_test': data.values[range(0, test_size)]}

np.save('./experiments/data/eicu.data',total)

data = np.load(path).item()