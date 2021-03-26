import numpy as np
import pandas as pd
import sklearn as skl
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers

node_feature_dim = 10
lookback_days = 50

historical_prices = pd.read_csv('data/historical_prices.csv')
sp500 = pd.read_csv('data/S&P500-Info.csv.csv')
symbols = sp500['Symbol']

'''
graph should only have 1s in upper triangular region
0 1 1
0 0 1
0 0 0
'''
def load_graph():
    with open('data/member_graph.npy', 'rb') as f:
        member_graph = np.load(f)
    return member_graph

def normalize_prices(prices):
    normalizer = skl.preprocessing.MinMaxScaler() # will normalize to [0,1] by default
    normalized_prices = normalizer.fit_transform(prices)
    return normalized_prices

def get_prices(index):    
    if index == 0:
        #columns = ['Adj Close','Close','High','Low','Open','Volume']
        columns = ['Adj Close', 'High','Low','Open','Volume']
    else:
        #columns = ['Adj Close' + '.' + str(index), 'Close' + '.' + str(index), 'High' + '.' + str(index), \
        #                'Low' + '.' + str(index), 'Open' + '.' + str(index),'Volume' + '.' + str(index)]
        columns = ['Adj Close' + '.' + str(index), 'High' + '.' + str(index), \
                        'Low' + '.' + str(index), 'Open' + '.' + str(index),'Volume' + '.' + str(index)]

    prices = historical_prices[columns]
    prices = prices.iloc[2:]
    np_prices = prices.values
    norm_prices = normalize_prices(np_prices)
    return norm_prices, np_prices

def get_training_data(node_num, norm_prices, prices):
    # copy() ?
    x_windows = np.array([norm_prices[i  : i + norm_prices] for i in range(len(norm_prices) - lookback_days)])
    
    # value of next day close for each x window, copy() ?
    next_day_close_values_norm = np.array([norm_prices[:,0][i + lookback_days] for i in range(len(norm_prices) - lookback_days)])
    next_day_close_values = np.array([prices[:,0][i + lookback_days] for i in range(len(prices) - lookback_days)])

    y_normalizer = skl.preprocessing.MinMaxScaler()
    y_normalizer.fit(np.expand_dims( next_day_close_values ))

    technical_indicators = []
    for x in x_windows:
        sma = np.mean(x[:,0]) # closing price
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)
		
    tech_ind_scaler = skl.preprocessing.MinMaxScaler()
    ti_norm = tech_ind_scaler.fit_transform(technical_indicators)

    assert x_windows.shape[0] == next_day_close_values_norm.shape[0] == ti_norm.shape[0]
    return x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm

def create_model(ti_shape):
    lstm_input = Input(shape=(lookback_days, 5), name='lstm_input')
    dense_input = Input(shape=(ti_shape,), name='tech_input')
    
    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)
    
    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # can add third branch for textual data
    
    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
    
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined) # save this for the node embedding
    z = Dense(1, activation="linear", name='dense_out')(z)
    
    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam,
                loss='mse')

# create f dimensional vector for each stock
def train_node(x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm):
    test_split = 0.9 # the percent of data to be used for testing
    n = int(x_windows.shape[0] * test_split)

    # splitting the dataset up into train and test sets
    x_train = x_windows[:n]
    ti_train = ti_norm[:n]
    y_train = next_day_close_values_norm[:n]

    x_test = x_windows[n:]
    ti_test = ti_norm[n:]
    y_test = next_day_close_values_norm[n:]

    unscaled_y_test = next_day_close_values[n:]
    model = create_model(ti_norm.shape[1])

    model.fit(x=[x_train, ti_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate([x_test, ti_test], y_test)


if __name__ == '__main__':
    member_graph = load_graph()
    dim = member_graph.shape[0]
    nodes = []
    row_sums = np.sum(member_graph, axis=1).tolist()
    col_sums = member_graph.sum(axis=0)
    historical_prices = pd.read_csv('data/historical_prices.csv')
    # if ith row or column has 1 node i is in the graph
    for i in range(dim):
        if row_sums[i] + col_sums[i] > 0:
            nodes.append(i)
            ticker = member_graph.iloc[i]['tickerLabel']
            norm_prices, prices = get_prices(i)
            x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm = get_training_data(i, norm_prices, prices)
            train_node(x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm)
    