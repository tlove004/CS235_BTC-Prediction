"""
By: Tyson Loveless
Given a labeled dataset and a model, trains the model on the dataset,
  and outputs training metrics recorded (accuracy, loss, etc.)

"""

from pandas import read_csv, concat, DataFrame
from util import parser, series_to_supervised
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from math import gcd
from build_models import build

'''
loads the feature data set and the label dataset, and trains the given model
 to find the given labels with the features
'''
def train(mdl, label_idx, keep_batch=True):
  
  dataset = read_csv('resources/datasets/all_features.csv', header=0, index_col=0, parse_dates=[0], date_parser=parser)
  labels = read_csv('resources/datasets/all_labels.csv', header=0, index_col=0, parse_dates=[0], date_parser=parser)
  
  # add our y vector to the dataset
  dataset = concat([dataset, labels.iloc[:, label_idx]], axis=1)
  #dataset.drop(dataset[:'2017-10-01'].index, inplace=True)
  DataFrame.dropna(dataset, inplace=True)
  #dataset = dataset[dataset.shape[0]-250000:]
  
  #labels = dataset[dataset.columns[-1]]
  #dataset.drop(dataset.columns[[-1]], axis=1, inplace=True)
  #dataset.drop(dataset.columns[5:15], axis=1, inplace=True)
  print(dataset.head())
  
  #dataset = dataset[:-1]
  #labels = labels[:-1]
  
  # split into test and train datasets
  
  values = dataset.values
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  num_features = scaled.shape[1]
  timesteps = 1
  reframed = series_to_supervised(scaled, timesteps, 1, True)
  print(reframed.columns)
  # drop all but last output value:
  reframed.drop(reframed.columns[[i for i in range(num_features*10, num_features*(timesteps+1)-2)]], axis=1, inplace=True)
  reframed = reframed[dataset.shape[0]-249950:]
  print(reframed.columns)
  percent_train = 0.83
  splitVal = int(reframed.shape[0] * percent_train)
  # inputs and outputs
  values = reframed.values
  train, train_y = values[:splitVal, :-2], values[:splitVal, -2:]
  test, test_y = values[splitVal:, :-2], values[splitVal:, -2:]
  train = train.reshape((train.shape[0], 1, train.shape[1]))
  test = test.reshape((test.shape[0], 1, test.shape[1]))
  # outputs
  #train_y = labels.values[:splitVal]
  #test_y = labels.values[splitVal:]
  #train_y = to_categorical(train_y, num_classes=3)
  #test_y = to_categorical(test_y, num_classes=3)
  
  print(test.shape, test_y.shape, train.shape, train_y.shape)
  
  batch_size = gcd(test.shape[0], train.shape[0])
  # batch_size = int(batch_size/10)
  
  print(batch_size)

  model = build(num_LSTM=1,
                batch_size=batch_size,
                num_features=test.shape[2],
                stateful=True,
                act=None,
                opt='adam',
                loss='mae')
  print(model.summary())

  callback = TensorBoard(log_dir='./graph',
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,)
  
  # train the model
  model.fit(train, train_y,
            epochs=2,
            batch_size=batch_size,
            validation_data=(test, test_y),
            verbose=1,
            shuffle=False,
            callbacks=[callback])
  
  # returns model that predicts in batches
  if keep_batch:
    return model
  
  # returns model that predicts online
  new_model = build(3, 1, test.shape[2], stateful=False, opt='adam', loss='mae')
  new_model.set_weights(model.get_weights())
  return new_model

mdl = 'model1.h5'
# mdl = 'model2.h5'
# mdl = 'model3.h5'
# mdl = 'model4.h5'
# mdl = 'model5.h5'

# we have 6 labels
model = train('resources/models/' + str(mdl), label_idx=0, keep_batch=False)

# save the model
model.save(mdl)
print(model.summary())
