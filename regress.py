"""
By: Tyson Loveless

This follows the lifecycle of a LSTM prediction model as found here:
 - https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/

 1.  Define Network   (hyperparameters set up)
 2.  Compile Network  (using hyperparameters and dataset size)
 3.  Fit Network      (train network to learn edge weights)
 4.  Make Predictions (we flip 4 and 5 as we switch from batch training to 1step predictions))
 5.  Evaluate Network (record error for each prediction at R^2 and RMSE)

As fitting the network takes a substantial amount of time, there are several models that
have already been fit, to perform batch prediction or 1-at-a-time prediction.
The training of models is currently commented out, while a load_model function loads
an already trained network from a given model file name.

Files are named: <n_batch>_<n_predictions>_<n_features>[_reshaped].h5
and can be found in the 'resources/models' directory
"""
from pandas import read_csv
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, LeakyReLU, Dense
from math import sqrt, gcd
from util import parser, reshape_model, prepare_data, invert_data, plot_predictions
import numpy as np


def train_new_model(train, n_batch, n_epochs, n_features, n_neurons):
    """
    creates, compiles, and fits an LSTM network for BTC prediction

    :param train: training dataset
    :param n_batch: number of instances to pass during an epoch iteration
    :param n_epochs: number of iterations that train on all instances
    :param n_features: number of input variables
    :param n_neurons: list of number of neurons for first 3 layers
    :return: the fitted model
    """
    # reshape training into [samples, timesteps, features]
    x, y = train[:, 0:n_features], train[:, n_features:]
    x = x.reshape(int(x.shape[0]), 1, x.shape[1])

    print("input shape:", x.shape)
    print("output shape:", y.shape)
    model = Sequential()
    model.add(
        LSTM(n_neurons[0], batch_input_shape=(n_batch, x.shape[1], x.shape[2]), return_sequences=True, stateful=False,
             dropout=0.05))  # , recurrent_dropout=0.5))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(LSTM(n_neurons[1], return_sequences=True, stateful=False, dropout=0.05))  # , recurrent_dropout=0.5))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(LSTM(n_neurons[2], return_sequences=False, stateful=False, dropout=0.05))  # , recurrent_dropout=0.5))
    #model.add(LeakyReLU(alpha=0.3))
    #model.add(LSTM(y.shape[1], return_sequences=False))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mae', optimizer='adamax')
    # fit network
    for i in range(n_epochs):
        print('~~~~~~~~~~~Epoch %d/%d~~~~~~~~~~~' % (i + 1, n_epochs))
        model.fit(x, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
    return model


def make_predictions(model, test, n_features, verbose=False):
    """
    Performs 1-at-a-time predictions on the test dataset

    :param model: a trained model, reshaped so that batch size is 1
    :param test: the test dataset
    :param n_features: number of inputs per instance
    :param verbose: if True, prints percentage completion
    :return: list of predictions on the given test features
    """
    mod = len(test) >> 6
    predictions = list()
    for i in range(len(test)):
        if verbose:
            if i % mod == 0:
                print("%d%%" % (int(i/len(test)*100)))
        # single step prediction -- get a single row
        x, y = test[i, 0:n_features], test[i, n_features:]
        prediction = model.predict(x.reshape(1, 1, len(x)), batch_size=1)
        predictions.append([p for p in prediction[0, :]])
    return predictions


def evaluate_model(ground_truth, predictions, n_predict):
    """
    Prints out the RMSE and R^2 values for the predictions against ground truth

    :param ground_truth: target values
    :param predictions: values our model returned
    :param n_predict: number of sequential predictions per time step
    :return: null
    """
    for i in range(n_predict):
        truth = [row[i] for row in ground_truth]
        predicted = [prediction[i] for prediction in predictions]
        rmse = sqrt(mean_squared_error(truth, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))
        r2 = r2_score(truth, predicted)
        print('t+%d R^2: %f' % ((i + 1), r2))


# load dataset
series = read_csv('resources/datasets/all_features.csv',
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)

# we'll be using single inputs, just the weighted price, for multivariate
# can keep more columns.  I did not see an improvement using other features
series.drop(series.columns[[i for i in range(0, 16)]], axis=1, inplace=True)
# leave some padding around our target (2017-1-1 : 2017-7-1) so that data prep doesn't cut out data
series = series['2016-8-1':'2017-12-1']
#series = series[0::60]  # if we want to subsample for example every hour (60 minutes)
print(series.head())

np.random.seed(40)  # for repeatability

# configure the hyper-parameters
percent_test = 0.5  # training/testing split percentage
beginning = '2017-1-1'
end = '2017-7-1'
n_lag = 50
n_features = series.shape[1] * n_lag
n_predict = 5
# 3 lstm layers with these numbers of neurons, a single output lstm layer with act activation function
n_neurons = [5, 5, 5]

# prepare data
print('preparing data')
y_scaler, train, test = prepare_data(series=series,
                                     n_lag=n_lag,
                                     n_predict=n_predict,
                                     beginning=beginning,
                                     end=end,
                                     percent_test=percent_test)

n_train, n_test = len(train), len(test)
series = series[beginning:end]

# configure fitting parameters
n_batch = gcd(n_test, n_train)
n_epochs = int(sqrt(n_batch))

'''
Uncomment this section and comment out the load_model below if desiring to create
a new model
'''
# prepares and fits a 4-layered LSTM model given the dataset and hyperparameters
print('setting up and training the model')
model = train_new_model(train=train,
                        n_batch=n_batch,
                        n_epochs=n_epochs,
                        n_features=n_features,
                        n_neurons=n_neurons)
# save model
model.save('resources/models/' + str(n_batch) + '_' + str(n_predict) + '_' + str(n_features) + '.h5')
# resize model for one-step predictions:
model = reshape_model(train=train,
                      n_features=n_features,
                      weights=model.get_weights(),
                      n_neurons=n_neurons)
# save re-sized model
model.save('resources/models/' + str(n_batch) + '_' + str(n_predict) + '_' + str(n_features) + '_reshaped.h5')

# print('loading model')
# model = load_model('resources/models/130320_30_5_reshaped.h5')

# make predictions
print('making %d predictions for each %d test instance' %(n_predict, n_test))
scaled_predictions = make_predictions(model=model,
                                      test=test,
                                      n_features=n_features,
                                      verbose=True)

# inverse transform predictions
print('inverting data')
predictions = invert_data(original=series,
                          scaled=scaled_predictions,
                          scaler=y_scaler,
                          n_test=(n_test + 2))

# inverse transform ground truth
scaled_ground_truth = [row[n_features:] for row in test]
ground_truth = invert_data(original=series,
                           scaled=scaled_ground_truth,
                           scaler=y_scaler,
                           n_test=(n_test + 2))


# perform evaluations
print('evaluating model')
evaluate_model(ground_truth, predictions, n_predict)
# np.savetxt('predictions.txt', predictions)

# visualize
print('preparing visualization')
plot_predictions(series=series,
                 predictions=predictions,
                 n_test=(n_test + 2),
                 n_predict=n_predict)
