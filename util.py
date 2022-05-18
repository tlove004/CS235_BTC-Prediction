"""
By: Tyson Loveless

Many of these utility functions were adapted from various tutorials, listed below.
Unless the reference is listed explicitly above the function, the function is either
entirely original, or more than 90% different, with perhaps naming
conventions maintained.

Tutorials and documentation referenced:
  For understanding how to set up a models in Keras:
 - https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
  Configuring data and preparing multivariate inputs:
 - https://pandas.pydata.org/pandas-docs/stable/api.html#dataframe
 - http://scikit-learn.org/stable/modules/preprocessing.html
 - https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
  For understanding nuances of LSTM models and multi-step prediction:
 - https://keras.io/models/sequential/
 - https://keras.io/layers/recurrent/#lstm
 - https://keras.io/optimizers/
 - https://keras.io/activations/
 - https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
 - https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
  .
"""

from pandas import to_datetime, concat, DataFrame
from keras.models import Sequential
from keras.layers import LSTM, LeakyReLU, Dense
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import numpy as np


def parser(x):
    """
    Returns x as a datetime object

    :param x: the date to parse
    :return: datetime object representing x
    """
    return to_datetime(x)


def show_all_plots(data, num_labels=0):
    """
    Plots the dataset and any associated class label frequencies

    :param data: The dataset to plot
    :param num_labels: Number of labels (for class labels)
    :return: pyplot of dataset broken into subplots
    """
    values = data.values
    groups = [i for i in range(0, len(values[0]) - num_labels)]
    pyplot.figure()
    i = 1
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(data.index, values[:, group], lw=0.5)
        pyplot.gca().axes.get_xaxis().set_visible(False)
        pyplot.title(data.columns[group], loc='right')
        i += 1

    pyplot.gca().axes.get_xaxis().set_visible(True)
    pyplot.xticks(rotation=40)
    pyplot.subplots_adjust(hspace=1.0, top=.95, bottom=0.1)

    if (num_labels > 0):
        pyplot.figure()
        k = 1
        for j in range(i, i + num_labels):
            pyplot.subplot(num_labels, 1, k)
            pyplot.hist(data.index, values[:, j - 1], s=10)
            pyplot.gca().axes.get_xaxis().set_visible(False)
            pyplot.title(data.columns[j - 1], loc='right')
            k += 1

        pyplot.gca().axes.get_xaxis().set_visible(True)
        pyplot.xticks(rotation=40)
        pyplot.subplots_adjust(hspace=1.0, top=.95, bottom=0.1)

    pyplot.show()


# ref: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
# minor changes to deal with multivariate inputs when doing multi-step predictions
def plot_predictions(series, predictions, n_test, n_predict):
    """
    Plots a given set of predictions on top of the given series

    :param series: ground truth values
    :param predictions: values predicted using regression model
    :param n_test: number of instances used in our predictions
    :param n_predict: number of sequential predictions at each step
    :return: plot of series and predictions
    """
    pyplot.plot(series[series.columns[[-1]]].values, color='blue', linewidth=1)
    # for each prediction in n_predict steps, plot on top of entire series in red
    for i in range(0, len(predictions), n_predict):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(predictions[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s][-1]] + predictions[i]
        pyplot.plot(xaxis, yaxis, color='red', linewidth=.65)
    # save and show the plot
    pyplot.savefig('plot.png')
    pyplot.show()


# ref: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
# minor changes to adjust for nuances of dataset.  As is did not work well for multivariate multi-step prediction
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, n_pred=1):
    """
    Frame a time series as a supervised learning dataset.

    Arguments:
      data: Sequence of observations as a list or NumPy array.
      n_in: Number of lag observations as input (X).
      n_out: Number of observations as output (y).
      dropnan: Boolean whether or not to drop rows with NaN values.
      n_pred: depth of lookback window
    Returns:
      Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[df.columns[-1]].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars - n_pred, n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars - n_pred, n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def reshape_model(train, n_features, weights, n_neurons):
    """
    Take a four-layered LSTM model (implicit input layer) with arbitrary batch input size
       and return a model with batch size 1 for performing 1-step predictions

    :param train: The training dataset
    :param n_features: How many features present per instance
    :param weights: The weights from the fitted model
    :param n_neurons: List of number of neurons per layer
    :return: The reshaped model
    """
    X, y = train[:, 0:n_features], train[:, n_features:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(n_neurons[0], batch_input_shape=(1, X.shape[1], X.shape[2]), return_sequences=True,
                   stateful=False, dropout=0.05))  # , recurrent_dropout=0.2))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(LSTM(n_neurons[1], return_sequences=True, stateful=False, dropout=0.05))  # , recurrent_dropout=0.1))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(LSTM(n_neurons[2], return_sequences=False, stateful=False, dropout=0.05))  # , recurrent_dropout=0.2))
    #model.add(LeakyReLU(alpha=0.3))
    #model.add(LSTM(y.shape[1], return_sequences=False))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mae', optimizer='adamax')
    model.set_weights(weights)
    return model


def prepare_data(series, n_lag, n_predict, beginning, end, percent_test):
    """
    Transforms a dataset into test and train datasets

    :param series: The dataset to transform
    :param n_lag: The number of steps in the past we want to include in the window
    :param n_predict: The number of sequential predictions to make
    :param beginning: A date string of the form '2017-1-1'
    :param end: A date string of the form '2017-7-1'
    :param percent_test: Float between (0, 1) for splitting into test/train sets
    :return: The scaler (transformation) object, train and test sets, and size of each of these sets
    """
    # dataset is non-stationary: augmented Dickey-Fuller test results in:
    # ADF Statistic: -0.416905
    # p - value: 0.907238
    # Critical Values:
    #   1 %: -3.430
    #   5 %: -2.862
    #   10 %: -2.567

    # make data stationary by taking the first difference
    diff_series = series.diff()
    diff_series.dropna(inplace=True)

    # save indexes for starting and stopping locations for our target range (beginning:end) and calculate n_test
    start = diff_series.index.get_loc(beginning).start
    stop = diff_series.index.get_loc(end).start
    total = stop - start
    n_test = int(total * percent_test)

    # scale between (-1, 1), using different scalers if multivariate data
    if series.shape[1] > 1:  # multivariate
        x = DataFrame(diff_series[diff_series.columns[:-1]].values.astype('float32'), index=diff_series.index)
        y = DataFrame(diff_series[diff_series.columns[-1:]].values.astype('float32'), index=diff_series.index)
        x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x.values)
        y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y.values)
        scaled_x = DataFrame(x_scaler.transform(x.values), index=x.index, columns=x.columns)
        scaled_y = DataFrame(y_scaler.transform(y.values), index=y.index, columns=[series.shape[1]-1])
        scaled = concat((scaled_x, scaled_y), axis=1)
    else:  # univariate
        diff_values = diff_series.values.astype('float32')
        y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(diff_series)
        scaled = y_scaler.transform(diff_values)

    # transform into supervised learning problem
    supervised = series_to_supervised(scaled, n_lag, n_predict)
    supervised = supervised[start:stop]
    supervised_values = supervised.values

    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return y_scaler, train, test


def invert_diff(prev, diff):
    """
    Inverts a differenced dataset given the first (undifferenced) value

    :param prev: original (undifferenced) value
    :param diff: a first-difference dataset
    :return: the inversion of a first differenced dataset
    """
    # invert the first item
    inverted = list()
    inverted.append(diff[0] + prev)
    # restore the rest of the values
    for i in range(1, len(diff)):
        inverted.append(diff[i] + inverted[i - 1])
    return inverted


def invert_data(original, scaled, scaler, n_test):
    """
    When preparing data, we took the first difference and ran a scaling function to make the data
    stationary and normalize it.  Here, we use the scaler object and invert the first difference
    to restore the original values.

    :return: the scaled data inverted to the original scale
    """
    inverted = list()
    for i in range(len(scaled)):
        # get scaled data from this row as a row vector
        invert = np.array(scaled[i]).reshape(1, len(scaled[i]))
        # invert the scale
        descaled = scaler.inverse_transform(invert)[0, :]
        # undo the difference, using previous instance at location idx
        idx = len(original) - n_test + i - 1
        prev = original.values[idx][-1]
        restored = invert_diff(prev, descaled)
        inverted.append(restored)
    return inverted
