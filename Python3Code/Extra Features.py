import sys
import copy
import pandas as pd
from pathlib import Path
import numpy as np

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction


df = pd.read_csv(r'C:/Users/MICK/Desktop/ML4QS/ML4QS/Python3Code/intermediate_datafiles/chapter4_result.csv')
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
df.index = df['Time']
df = df.drop(['Time'], axis=1)

DataViz = VisualizeDataset(__file__)

DataViz.plot_dataset(df, ['acc_phone_x_freq_0.8', 'label'], ['like', 'like'], ['line', 'points'])

