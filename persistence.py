"""
By: Tyson Loveless

Defines and plots a basic 1-step persistence model on the price of BTC

# details behind the persistence model found here:
# https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
"""


from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from util import parser

# details behind the persistence model found here:
# https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/

# read in data, drop all columns except weighted_price, and cut down range to first half of 2017
series = read_csv('resources/datasets/all_features.csv',
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)

series.drop(series.columns[[i for i in range(0, 16)]], axis=1, inplace=True)
series = series['2017-1-1':'2017-7-1']

# append shifted by 1 as the target
values = DataFrame(series.values)
df = concat([values.shift(1), values], axis=1)
df.columns = ['t', 't+1']

# get train and test sets
percent_test = 0.5
X = df.values
n_train = int(len(X) * (1 - 0.5))
train, test = X[1:n_train], X[n_train:]

# split into inputs and outputs for each set
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]

rmse = sqrt(mean_squared_error(test_y, test_X))
r2 = r2_score(test_y, test_X)

# inset plotting found
# https://stackoverflow.com/questions/13583153/how-to-zoomed-a-portion-of-image-and-insert-in-the-same-plot-in-matplotlib
fig = pyplot.figure()
ax1 = fig.add_subplot(111)
ax1.plot(train_y, label='actual', color='blue')

ax1.plot([None for i in train_y] + [x for x in test_y], linewidth=1, color='blue')
ax1.plot([None for i in train_y] + [x for x in test_X], label='predicted', linewidth=0.65, color='red')
pyplot.ylabel('BTC Value USD')
pyplot.title("Persistence Model ($R^2 = %f$, $RMSE = %f$)" % (r2, rmse))
pyplot.legend()

ax2 = pyplot.axes([.45, .5, .2, .2])
ax2.plot([None for i in train_y] + [x for x in test_y[3000:3015]], color='blue')
ax2.plot([None for i in train_y] + [x for x in test_X[3000:3015]], color='red')
pyplot.setp(ax2, xticks=[], yticks=[])

pyplot.show()
