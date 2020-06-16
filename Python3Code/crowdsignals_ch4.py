##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
from pathlib import Path

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/Assignment3/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_final.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter4_result.csv'

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# Compute the number of milliseconds covered by an instance based on the first two rows

# Chapter 4: Identifying aggregate attributes.

# First we focus on the time domain.

# Set the window sizes to the number of instances representing 2 minutes and 5 minutes

NumAbs = NumericalAbstraction()
#dataset_copy = copy.deepcopy(dataset)
#for ws in window_sizes:
#    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_x'], ws, 'mean')
#    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_x'], ws, 'std')
#    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_x'], ws, 'slope')

#DataViz.plot_dataset(dataset_copy, ['acc_x', 'acc_x_temp_mean', 'acc_x_temp_std', 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

ws = 4
selected_predictor_cols = ['heartrate','acc_x', 'acc_y', 'acc_z']
print('mean')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
print('std')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
print('slope')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'slope')

#DataViz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])


#CatAbs = CategoricalAbstraction()
#dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)

# Now we move to the frequency domain, with the same window size.

FreqAbs = FourierTransformation()
fs = float(1/30)

periodic_predictor_cols = ['heartrate','acc_x', 'acc_y', 'acc_z']
#data_table = FreqAbs.abstract_frequency(copy.deepcopy(dataset), ['acc_phone_x'], 4, fs)

# Spectral analysis.

#DataViz.plot_dataset(data_table, ['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])

dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, 4, fs)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

# The percentage of overlap we allow
window_overlap = 0.5
skip_points = int((1-window_overlap) * 4)
dataset = dataset.iloc[::skip_points,:]


dataset.to_csv(DATA_PATH / RESULT_FNAME)

DataViz.plot_dataset(dataset, ['heartrate', 'acc_x', 'acc_y', 'acc_z', 'label'],
                     ['like', 'like', 'like', 'like', 'like',],
                     ['line', 'line', 'line', 'line', 'points'])
