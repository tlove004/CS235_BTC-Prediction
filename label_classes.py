"""
By: Tyson Loveless

labels the dataset to frame the time series prediction problem as a classification problem
we perform several different labellings to test our models on consisting of the following classes:

-1: sell
0: hodl
1: buy

Different labellings at each timestep (minute) are given for the following

- minute to minute  (M+1)
- minute to 10 minutes  (M+10)
- minute to 30 minutes  (M+30)
- minute to 60 minutes  (M+60)
- minute to 120 minutes (M+120)
- minute to average confirmation (predict based upon a confirmation time lag)  (M+ACT)
"""

from pandas import read_csv, DateOffset, DataFrame
from math import ceil
from util import show_all_plots, parser


def get_label(p1, p2, scale):
    """
    given prices p1 and p2 (from specific time steps) and a scale,
      determines if buy/sell/hodl was the best choice
      and returns the class label

      @:param p1: price at time t
      @:param p2: price at time t+avg_completion
      @:param scale: the width between buy/sell/hold
      @:returns a label for the given prices
    """
    y = p2 - p1
    if y > (p1 * scale):
        return 1  # buy
    elif y < (p1 * -scale):
        return -1  # sell
    else:
        return 0  # hold





def label(x):
    """
    given a dataset containing high and low values as features,
      computes the average value for each time step and performs
      the various labellings described above

    @:param x: csv of features
    """
    # get data set
    data = read_csv(x, header=0, index_col=0, parse_dates=[0], date_parser=parser)

    data.rename(columns={'weighted_price': 'Average Price'}, inplace=True)

    avg = data[['Average Price']]
    conf = data[['avg_conf_time']]

    # to give a range for holding
    scale = 0.001

    # get labels for dataset
    last = data.index[-1]
    labels = DataFrame(index=data.index)
    times = [1, 10, 30, 60, 120]
    p1 = 0
    for i in data.index:
        try:
            # get price for this timestep
            p1 = avg.at[i, 'Average Price']

            # this timestep to set time prediction labels
            for time in times:
                next = i + DateOffset(minutes=time)
                if next <= last:
                    labels.at[(i, 'M+' + str(time))] = get_label(p1, avg.at[next, 'Average Price'], scale)

            next = i + DateOffset(minutes=ceil(conf.at[i, 'avg_conf_time']))
            if next <= last:
                # minute to average confirmation time predication labels
                labels.at[(i, 'M+ACT')] = get_label(p1, avg.at[next, 'Average Price'], scale)
        except Exception as e:
            print(i)
            print(p1)
            print(labels)
            print(str(e))

    return labels


labels = label('resources/datasets/all_features.csv')
labels.to_csv('resources/datasets/all_labels.csv')
show_all_plots(labels, num_labels=6)

exit(0)
