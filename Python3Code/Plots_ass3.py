from util.VisualizeDataset import VisualizeDataset
import pandas as pd

person_id = 1455390
DataViz = VisualizeDataset('assignment3.')

#CHAPTER 2
path = r'C:/Users/MICK/Desktop/ML4QS/ML4QS/Python3Code/intermediate_datafiles/Assignment3/'
df = pd.read_csv(path+'chapter2_result.csv')

df.index = pd.to_datetime(df['time'])
df = df[df['personid'] == person_id]

DataViz.plot_dataset(df, ['acc_x', 'acc_y', 'acc_z', 'heartrate', 'heartrate_std', 'label'],
                         ['like', 'like', 'like', 'exact', 'exact', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'points'])

#CHAPTER 3
path = r'C:/Users/MICK/Desktop/ML4QS/ML4QS/Python3Code/intermediate_datafiles/Assignment3/'
df = pd.read_csv(path+'chapter3_result_final.csv')

df.index = pd.to_datetime(df['time'])
df = df[df['personid'] == person_id]

DataViz.plot_dataset(df, ['acc_x', 'acc_y', 'acc_z', 'heartrate', 'pca_', 'label'],
                              ['like', 'like', 'like', 'like', 'like', 'like'],
                              ['line', 'line', 'line', 'line', 'line', 'points'])

print(df.columns)

#CHAPTER 4
path = r'C:/Users/MICK/Desktop/ML4QS/ML4QS/Python3Code/intermediate_datafiles/Assignment3/'
df = pd.read_csv(path+'chapter4_result.csv')

df.index = pd.to_datetime(df['time'])
df = df[df['personid'] == person_id]

DataViz.plot_dataset(df, ['acc_x_max_freq', 'acc_x_freq_weighted', 'acc_x_pse', 'acc_x_freq_0.017_Hz_ws_6', 'label'],
                            ['like', 'like', 'like', 'like', 'like'],
                            ['line', 'line', 'line', 'line', 'points'])

DataViz.plot_dataset(df, ['heartrate_temp_mean_ws_6', 'heartrate_temp_std_ws_6', 'heartrate_temp_slope_ws_6', 'label'],
                             ['exact', 'exact', 'exact', 'like'],
                             ['line', 'line', 'line', 'points'])
