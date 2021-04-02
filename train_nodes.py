import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam

node_feature_dim = 10
lookback_days = 50

historical_prices = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/historical_prices.csv')
hist_symbols = historical_prices.iloc[[0]].to_numpy().squeeze()
sp500 = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/S&P500-Info.csv')
symbols = sp500['Symbol']

'''
graph should only have 1s in upper triangular region
0 1 1
0 0 1
0 0 0
'''
def load_graph():
    with open('/Users/liam_adams/my_repos/finance_gnn/data/member_graph.npy', 'rb') as f:
        member_graph = np.load(f)
    return member_graph

def normalize_prices(prices):
    normalizer = MinMaxScaler() # will normalize to [0,1] by default
    normalized_prices = normalizer.fit_transform(prices) # scales each column independently
    return normalized_prices

def get_prices(ticker):    
    col_inds = np.where(hist_symbols == ticker)[0]
    col_name = historical_prices.iloc[:, col_inds[0]].name
    val = col_name.split('.')
    if len(val) == 1:
        index = 0
    else:
        index = val[1]
    if index == 0:
        #columns = ['Adj Close','Close','High','Low','Open','Volume']
        columns = ['Adj Close', 'High','Low','Open','Volume']
    else:

        #columns = ['Adj Close' + '.' + str(index), 'Close' + '.' + str(index), 'High' + '.' + str(index), \
        #                'Low' + '.' + str(index), 'Open' + '.' + str(index),'Volume' + '.' + str(index)]
        columns = ['Adj Close' + '.' + str(index), 'High' + '.' + str(index), \
                        'Low' + '.' + str(index), 'Open' + '.' + str(index),'Volume' + '.' + str(index)]

    prices = historical_prices[columns]
    prices = prices.iloc[2:] # first 2 rows aren't prices
    np_prices = prices.to_numpy()
    norm_prices = normalize_prices(np_prices)
    return norm_prices, np_prices

def get_training_data(node_num, norm_prices, prices):
    # changes to slices will reflect in original so use copy
    x_windows = np.array([norm_prices[i  : i + lookback_days].copy() for i in range(len(norm_prices) - lookback_days)])
    
    # value of next day close for each x window, predicting next day close
    next_day_close_values_norm = np.array([norm_prices[:,0][i + lookback_days].copy() for i in range(len(norm_prices) - lookback_days)])
    next_day_close_values_norm = np.expand_dims(next_day_close_values_norm, -1) # make 2D
    
    next_day_close_values = np.array([prices[:,0][i + lookback_days] for i in range(len(prices) - lookback_days)])
    # expand_dims?

    y_normalizer = MinMaxScaler()
    #y_normalizer.fit(np.expand_dims( next_day_close_values )) # allows us to un normalize at the end

    technical_indicators = []
    for x in x_windows:
        sma = np.mean(x[:,0]) # get average of closing price of each window
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)
		
    tech_ind_scaler = MinMaxScaler()
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

    adam = Adam(lr=0.0005)

    model.compile(optimizer=adam,
                loss='mse')

    return model

# create f dimensional vector for each stock
# need to split training and test data before normalizing, use fit_transform on training data, transform on test data
# minmaxscaler remembers mean and variance from fit_transform to scale test data accordingly, fit_transform first
# also need to handle NANs if any
def train_node(x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm, symbol):
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

    # make sure validation data is newer than training data
    model.fit(x=[x_train, ti_train], y=y_train, batch_size=32, epochs=10, shuffle=False, validation_split=0.1)
    evaluation = model.evaluate([x_test, ti_test], y_test)
    model.save_weights('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + symbol)


if __name__ == '__main__':
    member_graph = load_graph()
    dim = member_graph.shape[0] # comes from members.csv, symbols in alhpabetical order
    print('nodes in graph', dim)
    nodes = []
    row_sums = np.sum(member_graph, axis=1).tolist()
    col_sums = member_graph.sum(axis=0)
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')
    # if ith row or column has a 1, node i is in the graph
    for i in range(dim):
        if row_sums[i] + col_sums[i] > 0:
            nodes.append(i)
            # get ticker label from members.csv
            ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0]
            print('training symbol', ticker)
            norm_prices, prices = get_prices(ticker)
            x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm = get_training_data(i, norm_prices, prices)
            train_node(x_windows, next_day_close_values_norm, next_day_close_values, y_normalizer, ti_norm, ticker)
    