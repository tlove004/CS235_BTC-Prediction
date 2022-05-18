"""
By: Tyson Loveless
takes in the average time for confirmation for a bitcoin every 12 hours
and interpolates between minutes using a cubic fit
outputs to avg-conf-updated.csv
This information is used to generate labels for buy/sell/hold given a weighted price at timestep t
"""

from pandas import read_csv, to_datetime, date_range
import numpy as np
from util import show_all_plots, parser

# read in average confirmation time dataset
data = read_csv('avg-confirmation-time.csv', header=None, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# get interpolation rows
fill = to_datetime(date_range('2016-4-20', '2018-3-27', freq='60S'))

# store minimum nonzero element
minimum = min(x for x in data if x > 0)

# filter outliers above 1/4 standard deviations beyond mean
data = data[data-np.mean(data) <= 0.25 * np.std(data)]

# store first and last element
first = data[0]
last = data[-1]

# add interpolation rows
data = data.reindex(fill, fill_value=None)

# restore first and last values in case reindexing overwrote them
data[0] = first
data[-1] = last

# interpolate using cubic fit
data = data.interpolate(method='cubic')

# values <= 0 set to previous minimum
data[data <= 0] = minimum

# save output
data.to_csv('avg-conf-updated.csv')

# plot data set
show_all_plots(data)
