import sys
import pandas
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import pydot


def parser(x):
    return pandas.to_datetime(x, unit='s')

file_name = sys.argv[1]
# Random Forest
series = read_csv(file_name, date_parser=parser)

# create an array from price and features
labels = np.array(series['price'])
series = series.drop('price', axis=1)
feature_list = list(series.columns)
series = np.array(series)

# split data in half, train first part, and predict the second
train_features = []
test_features = []
train_labels = []
test_labels = []

for i in range(len(series)):
    if i < len(series)/2:
        train_features.append(series[i])
        train_labels.append(labels[i])
    else:
        test_features.append(series[i])
        test_labels.append(labels[i])

# sample the training and test sets to get half an hour granularity
train_features_sample = []
test_features_sample = []
train_labels_sample = []
test_labels_sample = []
main_prediction = []

for i in range(len(train_features)):
    if i % 30 == 0:
        train_features_sample.append(train_features[i])
        train_labels_sample.append(train_labels[i])

for i in range(len(test_features)):
    if i % 30 == 0:
        test_features_sample.append(test_features[i])
        test_labels_sample.append(test_labels[i])

# output a png file for a simple tree in random forest
rf = RandomForestRegressor(n_estimators=1000, random_state=42, warm_start=True, n_jobs=4, max_depth=3)
model = rf.fit(train_features_sample, train_labels_sample)
tree_small = rf.estimators_[5]
# Save the tree as a png image
tree.export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')


# print importance scores
print(model.feature_importances_)
# plot importance scores
names = feature_list
print(names)
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, model.feature_importances_)
pyplot.xticks(ticks, names, rotation='60')
pyplot.show()

# in each step, we train a model, predict 100 data points, and then retrain the model
t = 0
print("random Forest Prediction:")
while True:
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, warm_start=True, n_jobs=4)
    # Train the model on training data

    model = rf.fit(train_features_sample, train_labels_sample)

    tfs = []
    for i in range(100):
        if t + i < len(test_labels_sample):
            tfs.append(test_features_sample[t + i])
            # Use the forest's predict method on the test data

    prediction = rf.predict(tfs)
    for i in range(len(prediction)):
        print(prediction[i])

    main_prediction.append(prediction)

    for i in range(100):
        if t + i < len(test_labels_sample):
            train_features_sample.append(test_features_sample[t + i])
            train_labels_sample.append(test_labels_sample[t + i])

    t += 100
    if t > len(test_labels_sample):
        break

'''
# Calculate the absolute errors
pyplot.plot(test_labels_sample)
pyplot.plot(main_prediction, color='red')
pyplot.show()

errors = abs(main_prediction - test_labels_sample)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

error = mean_squared_error(test_labels_sample, main_prediction)

print('Test MSE: %.3f' % error)

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels_sample)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

dates = series[:, feature_list.index('Timestamp')]
conv_dates = []

for t in range(len(dates)):
    conv_dates.append(parser(dates[t]))

for d in conv_dates:
    print(d)

true_data = DataFrame(data={'date': conv_dates, 'actual': labels})

dates = test_features[:, feature_list.index('Timestamp')]

conv_dates = []
for t in range(len(dates)):
    conv_dates.append(parser(dates[t]))
for d in conv_dates:
    print(d)
predictions_data = DataFrame(data={'date': conv_dates, 'prediction': predictions})

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()
# Graph labels
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Actual and Predicted Values')
'''

# Arima
fields = ['Timestamp', 'price']
series = read_csv(file_name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=fields)
X = series.values

train = []
test = []

for i in range(len(X)):
    if i < len(X)/2:
        train.append(X)
    else:
        test.append(X)

sample_train = []
sample_test = []

for i in range(len(X)/2):
    if i % 60 == 0:
        sample_train.append(train[i])

for i in range(len(test)):
    if i % 60 == 0:
        sample_test.append(test[i])

history = [x for x in sample_train]

predictions = list()

print ("ARMIA Predictions:")

for t in range(len(sample_test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = sample_test[t]
    history.pop(0)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(sample_test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(sample_test)
pyplot.plot(predictions, color='red')
pyplot.show()

print(series.head())
series.plot()
autocorrelation_plot(series)
pyplot.show()

# fit model
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
