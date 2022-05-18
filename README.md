--------------------------------------------------------------------------------

- Bitcoin Price Prediction
- Tyson Loveless
- Farzin Houshmand
- Amin Kalantar

--------------------------------------------------------------------------------

# For running ARIMA and Random Forest demos

```
>python3 RFArima.py Filename.csv
```

Where `filename.csv` is the dataset

This demo outputs the prediction results for ARIMA and Random Forest models, plots a graph for each of them, plots a sample binary decision tree based on the trained Random Forest model, and calculates the error for each model.

# For running the persistence model demo:

```
>python3 persistence.py
```

This demo will output a graph of BTC price from 2017-1-1 to 2017-7-1 with the persistence value starting from 2017-4-1 through 2017-7-1, and a zoomed in portion displayed to show the effect of the persistence "dummy" model.

# For running the LSTM model demo:

```
>python3 lstm.py
```

This demo loads a trained LSTM model that uses only price as a feature, with a 50 time-step lag window as input (vector of 50 values) with a five time-step output.

Several trained models are available to load in the `resources/model/` directory. Refer to comments for setting hyper-parameters on other models.

To train a new model, hyper-parameters `n_lag`, `n_predict`, and `n_neurons` could be set and the training/loading section's comments could be toggled.

# For running the Bayesian NN model demo:

```
>python3 BNN.py Filename.csv
```

Where `filename.csv` is the dataset

This demo will output a graph of BTC price from 2017-1-1 to 2017-7-1 with the persistence value starting from 2017-4-1 through 2017-7-1\. The graph also contains a simple SVR and dummy model as the baselines.

## Requirements:

The following packages must be installed

- h5py
- Keras
- matplotlib
- numpy
- pandas
- pydot
- python-dateutil
- scikit-learn
- scipy
- statsmodels
- tensorboard
- tensorflow
- edward
- statsmodels
