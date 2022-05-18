import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import edward as ed
from edward.models import Normal
from sklearn.metrics import r2_score
from scipy.ndimage.interpolation import shift
import sys

tf.flags.DEFINE_integer("N", default=2172, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=11, help="Number of features.")
tf.flags.DEFINE_float("R", default=0.5, help="test to train ratio")

FLAGS = tf.flags.FLAGS

def evaluate_bnn(X_train, X_test, y_train, y_test):

    ##Linear SVR
    from sklearn.svm import SVR
    # Train using a linear kernel
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Linear Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    

    ##dummy
    
    y_pred_dummy = shift(y_test, 60)
    r_2_bnn = r2_score(y_test, y_pred_dummy)
    yield 'Dummy ($R^2={:.3f}$)'.format(r_2_bnn), y_test, y_pred_dummy


    ##ARIMA
    #y_pred_rand = pd.read_csv("forest-pred.csv", header=None)
    #y_pred_rand = np.array(y_pred_rand, dtype=np.float32)
    #y_pred_rand = MinMaxScaler().fit_transform(y_pred_rand)
    #y_pred_rand = y_pred_rand[:, -1]
    
    #y_test_rand = pd.read_csv("forest-exp.csv", header=None)
    #y_test_rand = np.array(y_test_rand, dtype=np.float32)
    #y_test_rand = MinMaxScaler().fit_transform(y_test_rand)
    #y_test_rand = y_test_rand[:, -1]
    #r_2_bnn = r2_score(y_test_rand, y_pred_rand)
    #yield 'ARIMA ($R^2={:.3f}$)'.format(r_2_bnn), y_test_rand, y_pred_rand


    ##ARIMA
    
    #y_pred_rand = pd.read_csv("randomforest-pred.csv", header=None)
    #y_pred_rand = np.array(y_pred_rand, dtype=np.float32)
    #y_pred_rand = MinMaxScaler().fit_transform(y_pred_rand)
    #y_pred_rand = y_pred_rand[::2, -1]
    
    #y_test_rand = pd.read_csv("forest-exp.csv", header=None)
    #y_test_rand = np.array(y_test_rand, dtype=np.float32)
    #y_test_rand = MinMaxScaler().fit_transform(y_test_rand)
    #y_test_rand = y_test_rand[:, -1]
    #r_2_bnn = r2_score(y_test_rand, y_pred_rand)
    #yield 'Random Forest ($R^2={:.3f}$)'.format(r_2_bnn), y_test_rand, y_pred_rand


    ##LSTM
    #y_pred_rand = pd.read_csv("one_step_lstm", header=None)
    #y_pred_rand = y_pred_rand[::30]
    #y_pred_rand = np.array(y_pred_rand, dtype=np.float32)
    #y_pred_rand = MinMaxScaler().fit_transform(y_pred_rand)
    #y_pred_rand = y_pred_rand[:, -1]
    
    #y_test_rand = pd.read_csv("forest-exp.csv", header=None)
    #y_test_rand = np.array(y_test_rand, dtype=np.float32)
    #y_test_rand = MinMaxScaler().fit_transform(y_test_rand)
    #y_test_rand = y_test_rand[:, -1]
    #r_2_bnn = r2_score(y_test, y_pred_rand)
    #yield 'LSTM ($R^2={:.3f}$)'.format(0.942), y_test, y_pred_rand



    ##BNN
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 20]), scale=tf.ones([FLAGS.D, 20]),
                   name="W_0")
    W_1 = Normal(loc=tf.zeros([20, 20]), scale=tf.ones([20, 20]), name="W_1")
    W_2 = Normal(loc=tf.zeros([20, 1]), scale=tf.ones([20, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(20), scale=tf.ones(20), name="b_0")
    b_1 = Normal(loc=tf.zeros(20), scale=tf.ones(20), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X, W_0,W_1,W_2, b_0,b_1,b_2), scale=0.1 * tf.ones(FLAGS.N), name="y")

    with tf.variable_scope("posterior"):
      with tf.variable_scope("qW_0"):
        loc = tf.get_variable("loc", [FLAGS.D, 20])
        scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, 20]))
        qW_0 = Normal(loc=loc, scale=scale)
      with tf.variable_scope("qW_1"):
        loc = tf.get_variable("loc", [20, 20])
        scale = tf.nn.softplus(tf.get_variable("scale", [20, 20]))
        qW_1 = Normal(loc=loc, scale=scale)
      with tf.variable_scope("qW_2"):
        loc = tf.get_variable("loc", [20, 1])
        scale = tf.nn.softplus(tf.get_variable("scale", [20, 1]))
        qW_2 = Normal(loc=loc, scale=scale)
      with tf.variable_scope("qb_0"):
        loc = tf.get_variable("loc", [20])
        scale = tf.nn.softplus(tf.get_variable("scale", [20]))
        qb_0 = Normal(loc=loc, scale=scale)
      with tf.variable_scope("qb_1"):
        loc = tf.get_variable("loc", [20])
        scale = tf.nn.softplus(tf.get_variable("scale", [20]))
        qb_1 = Normal(loc=loc, scale=scale)
      with tf.variable_scope("qb_2"):
        loc = tf.get_variable("loc", [1])
        scale = tf.nn.softplus(tf.get_variable("scale", [1]))
        qb_2 = Normal(loc=loc, scale=scale)

    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1,
                         W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
    inference.run(n_iter=2000, n_samples=128)
    
    out = neural_network(X_test, qW_0.sample(), qW_1.sample(), qW_2.sample(), qb_0.sample(), qb_1.sample(),qb_2.sample())
   # pd.DataFrame(out.eval()).plot()
    bnn_pred = out.eval()
    r_2_bnn = r2_score(y_test, bnn_pred)
    yield 'BNN($R^2={:.3f}$)'.format(r_2_bnn), y_test, bnn_pred


def plot(results,name, sbnn):


    # Using subplots to display the results on the same X axis
    fig, plts = plt.subplots(nrows=len(results), figsize=(6, 12))
    fig.canvas.set_window_title('Predicting data from ')
    # Show each element in the plots returned from plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        subplot.set_xticklabels(())
        subplot.set_yticklabels(())


        # Label the vertical axis
        subplot.set_ylabel('stock price')

        # Set the title for the subplot
        subplot.set_title(title)

        # Plot the actual data and the prediction
        subplot.plot(y, 'b', label='actual')
        if not str(title).startswith("LSTM"):
            mid=len(y_pred) // 2
            y_pred = y_pred[mid:]
            x_new = np.arange(mid, len(y))
            subplot.plot(x_new, y_pred, 'r', label='predicted')
        else:
            mid=len(y) // 2
            x_new = np.arange(mid, len(y))
            y_pred=y_pred[::2]
            subplot.plot(x_new, y_pred, 'r', label='predicted')
       

        # Mark the extent of the training data
        subplot.axvline(len(y) // 2, linestyle='--', color='0', alpha=0.2)

        # Include a legend in each subplot
        subplot.legend()

    # Let matplotlib handle the subplot layout
    fig.tight_layout()

    # To save the plot to an image file, use savefig()
    plt.savefig(str(name)+'.png')

    plt.show()

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()

def neural_network(X,W_0,W_1,W_2,b_0,b_1,b_2):
  h = tf.sigmoid(tf.matmul(X, W_0) + b_0)
  h = tf.sigmoid(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 50)
    cax = ax1.imshow(df, interpolation='nearest', cmap=cmap)
    #ax1.grid(True)
    plt.title('Bictoin Price Feature Correlation')
    labels=['Market Cap','Avg Block Size','Hash Rate','Difficulty','Bitcoin Market Cap %','Bitcoin Cash Market Cap %','Ripple Market Cap %','Ethereum Market Cap %','Dash Market Cap %','Litecoin Market Cap %', 'Others Market Cap %',]
    ax1.set_xticklabels(())
    ax1.set_yticklabels(())
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1])
    plt.savefig('corr.png')
    plt.show()


if __name__ == '__main__':

    mmscalar = MinMaxScaler()
    sscalar = StandardScaler()

    filename = sys.argv[1]
    frame = pd.read_csv(filename, header=None)
    frame = frame[[5,6,7,8,9,10,11,12,13,14,15,1]]
    secondframe = frame.copy()


    #frame.corr().to_csv('corr.csv')
    #correlation_matrix(frame.corr()[[1]][0:11])
    
    #print(frame.corr())
    
    frame[1] = frame[1].shift(60)

    frame = frame[310321:570961:60]
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))

    arr = np.array(frame, dtype=np.float32)
    
    arr = mmscalar.fit_transform(arr)

    X, y = arr[:, :-1], arr[:, -1]

    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=FLAGS.R)
    X_test, y_test = X, y

    scaler = sscalar

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Evaluating regression learners")

    results = list(evaluate_bnn(X_train,X_test,y_train,y_test))

    # Display the results
    print("Plotting the results")
    plot(results, 'shifted', sscalar)