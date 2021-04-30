from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import params

#sp500 = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/S&P500-Info.csv')
#symbols = sp500['Symbol']

def normalize_prices(prices):
    normalizer = MinMaxScaler() # will normalize to [0,1] by default
    normalized_prices = normalizer.fit_transform(prices) # scales each column independently
    return normalized_prices

def get_prices(ticker, historical_prices, hist_symbols):
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
    prices = prices.fillna(method='ffill')
    np_prices = prices.to_numpy()
    #norm_prices = normalize_prices(np_prices)
    return np_prices

def get_regression_training_data():
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')
    historical_prices = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/historical_prices.csv') 
    hist_symbols = historical_prices.iloc[[0]].to_numpy().squeeze()
    dim = members.shape[0]

    neighbor_map = {}
    x_train_windows_all = []
    x_test_windows_all = []
    y_train_norm_all = []
    y_train_values_all = []
    y_train_normalizer_all = []
    ti_train_norm_all = []
    ti_test_norm_all = []
    y_test_norm_all = []
    y_test_values_all = []
    symbols = []

    for i in range(dim):
        # this selects the row, not affected by the deletion
        ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0] # stocks in alphabetical order in member_graph and members
        symbols.append(ticker)
        prices = get_prices(ticker, historical_prices, hist_symbols)
        #node_map[i] = ticker
        #neighbors = get_neighbors(member_graph, i) # returns row inds, use tf embedding lookup to lookup node embeddings
        #neighbor_map[i] = neighbors
        test_split = 0.9 # the percent of data to be used for testing
        n = int(prices.shape[0] * test_split)
        x_train = prices[:n]
        x_train_norm = normalize_prices(x_train)
        x_test = prices[n:]
        x_test_norm = normalize_prices(x_test)
        # changes to slices will reflect in original so use copy, i + params.lookback_days not included
        x_train_windows = np.array([x_train_norm[i  : i + params.lookback_days].copy() for i in range(len(x_train_norm) - params.lookback_days)])
        x_train_windows_all.append(x_train_windows)
        x_test_windows = np.array([x_test_norm[i  : i + params.lookback_days].copy() for i in range(len(x_test_norm) - params.lookback_days)])
        x_test_windows_all.append(x_test_windows)
        
        # value of next day close for each x window, predicting next day close
        next_day_close_values_norm = np.array([x_train_norm[:,0][i + params.lookback_days].copy() for i in range(len(x_train_norm) - params.lookback_days)])
        next_day_close_values_norm = np.expand_dims(next_day_close_values_norm, -1) # make 2D
        y_train_norm = next_day_close_values_norm
        y_train_norm_all.append(y_train_norm)

        next_day_close_values = np.array([x_train[:,0][i + params.lookback_days] for i in range(len(x_train) - params.lookback_days)])
        # expand_dims?
        y_train_values = next_day_close_values
        y_train_values_all.append(y_train_values)

        y_train_normalizer = MinMaxScaler() # this remembers original data
        y_train_normalizer.fit(np.expand_dims( y_train_values, -1 )) # allows us to un normalize at the end
        y_train_normalizer_all.append(y_train_normalizer)

        test_next_day_close_values_norm = np.array([x_test_norm[:,0][i + params.lookback_days].copy() for i in range(len(x_test_norm) - params.lookback_days)])
        test_next_day_close_values_norm = np.expand_dims(test_next_day_close_values_norm, -1) # make 2D
        y_test_norm = test_next_day_close_values_norm
        y_test_norm_all.append(y_test_norm)

        test_next_day_close_values = np.array([x_test[:,0][i + params.lookback_days] for i in range(len(x_test) - params.lookback_days)])
        # expand_dims?
        y_test_values = test_next_day_close_values
        y_test_values_all.append(y_test_values)
        
        ti_train = []
        for x in x_train_windows:
            sma = np.mean(x[:,0]) # get average of closing price of each window
            ti_train.append(np.array([sma]))

        ti_train = np.array(ti_train)            

        ti_test = []
        for x in x_test_windows:
            sma = np.mean(x[:,0]) # get average of closing price of each window
            ti_test.append(np.array([sma]))

        ti_test = np.array(ti_test)
            
        tech_ind_scaler = MinMaxScaler()
        ti_train_norm = tech_ind_scaler.fit_transform(ti_train)
        ti_train_norm_all.append(ti_train_norm)

        tech_ind_scaler = MinMaxScaler()
        ti_test_norm = tech_ind_scaler.fit_transform(ti_test)
        ti_test_norm_all.append(ti_test_norm)
    
    x_train_windows_all = np.stack(x_train_windows_all)
    x_test_windows_all = np.stack(x_test_windows_all)
    y_train_norm_all = np.stack(y_train_norm_all)
    y_train_values_all = np.stack(y_train_values_all)
    y_train_normalizer_all = np.stack(y_train_normalizer_all) # this remembers normalizations made to each stock and can undo them
    ti_train_norm_all = np.stack(ti_train_norm_all)
    ti_test_norm_all = np.stack(ti_test_norm_all)
    y_test_norm_all = np.stack(y_test_norm_all)
    y_test_values_all = np.stack(y_test_values_all)

    assert x_train_windows_all.shape[0] == y_train_norm_all.shape[0] == ti_train_norm_all.shape[0]
    return x_train_windows_all, x_test_windows_all, y_train_norm_all, y_train_values_all, y_train_normalizer_all, \
                    ti_train_norm_all, ti_test_norm_all, y_test_norm_all, y_test_values_all, neighbor_map, symbols