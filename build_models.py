"""
By: Tyson Loveless
Defines and saves several LSTM models for testing

Used keras documentation here: https://keras.io/getting-started/sequential-model-guide/
   for understanding how to build models
"""

from pandas import read_csv
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from util import parser
from keras.utils import plot_model

#data = read_csv('resources/datasets/all_features.csv', header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)

# get input and output shape for models

#data_dim = data.shape[1]
#timesteps = 1  # this is the window size
#batch_size = 1

def build(num_LSTM, batch_size, num_features, num_out=1, timesteps=1, num_nodes=32, stateful=True, act='softmax', opt='nadam', loss='categorical_crossentropy'):
  print('building model')
  model = Sequential()
  #size = batch_size if batch else 1
  model.add(LSTM(num_nodes, input_shape=(timesteps, num_features), batch_size=batch_size,
                 return_sequences=False, stateful=stateful, recurrent_dropout=0.2))
  #model.add(Dropout(0.2))
  # for i in range (2, num_LSTM):
  #   model.add(LSTM(num_nodes, return_sequences=True, stateful=batch))
  #   model.add(Dropout(0.2))
  #model.add(LSTM(num_nodes, stateful=batch))
  #model.add(Dropout(0.2))
  #model.add(Dense(num_out, activation=act))
  model.add(Dense(2, activation=act))
  model.compile(loss=loss, optimizer=opt)
  return model


# set up our first test model
if __name__ == "__main__":
  model = build(num_LSTM=3,
                batch_size=5000,
                num_features=2,
                num_nodes=32,
                stateful=True,
                act='softmax',
                opt='nadam')
  model.save("resources/models/model1.h5")
  plot_model(model, to_file='resources/models/model1.png')