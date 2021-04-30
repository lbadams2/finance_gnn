from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import params

#sp500 = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/S&P500-Info.csv')
#symbols = sp500['Symbol']

'''
graph should only have 1s in upper triangular region
0 1 1
0 0 1
0 0 0
'''
def load_graph():
    with open('/Users/liam_adams/my_repos/finance_gnn/data/member_graph.npy', 'rb') as f:
        member_graph = np.load(f)
    member_graph = np.maximum(member_graph, member_graph.transpose()) # make matrix symmetric
    return member_graph

def load_embeddings(members, train=True):
    dim = members.shape[0]
    all_embeddings = []
    for i in range(dim):
        # this selects the row, not affected by the deletion
        ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0]
        if train:
            with open('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + ticker + '.npy', 'rb') as f:
                embedding = np.load(f)
                all_embeddings.append(embedding)
        else:
            with open('/Users/liam_adams/my_repos/finance_gnn/embeddings/test/' + ticker + '.npy', 'rb') as f:
                embedding = np.load(f)
                all_embeddings.append(embedding)
    all_embeddings = np.stack(all_embeddings)
    return all_embeddings

def delete_nodes(members, member_graph, y_train):
    row_sums = np.sum(member_graph, axis=1).tolist()
    nodes_to_delete = []
    node_count = 0
    dim = members.shape[0]
    for i in range(dim):
        if row_sums[i] < 1:
            nodes_to_delete.append(i)

    member_graph = np.delete(member_graph, nodes_to_delete, axis=0) # delete nodes with no edges
    member_graph = np.delete(member_graph, nodes_to_delete, axis=1) # delete nodes with no edges
    #embeddings = np.delete(embeddings, nodes_to_delete, axis=0)
    y_train = np.delete(y_train, nodes_to_delete, axis=0)
    
    # need to also delete from members
    members = members.drop(nodes_to_delete)
    return members, member_graph, y_train

def get_neighbors(graph, symbols):
    neighbor_inds = {}
    for i, symbol in enumerate(symbols):
        row = graph[i]
        col = graph[ : , i]
        row_inds = np.where(row == 1)[0]
        neighbor_inds[i] = row_inds
        #col_inds = np.where(col == 1)[0]
        #all_inds = np.concatenate((row_inds, col_inds))
        #return all_inds
    return neighbor_inds

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
    member_graph = load_graph()
    assert members.shape[0] == member_graph.shape[0]

    x_train_windows_all = []
    x_test_windows_all = []
    y_train_norm_all = []
    y_train_values_all = []
    y_train_normalizer_all = []
    ti_train_norm_all = []
    ti_test_norm_all = []
    y_test_norm_all = []
    y_test_normalizer_all = []
    y_test_values_all = []
    symbols = []

    dim = members.shape[0]
    for i in range(dim):
        # this selects the row, not affected by the deletion
        ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0] # stocks in alphabetical order in member_graph and members
        symbols.append(ticker)
        prices = get_prices(ticker, historical_prices, hist_symbols)
        #node_map[i] = ticker
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

        y_test_normalizer = MinMaxScaler()
        y_test_normalizer.fit(np.expand_dims(y_test_values,-1))
        y_test_normalizer_all.append(y_test_normalizer)
        
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
                    ti_train_norm_all, ti_test_norm_all, y_test_norm_all, y_test_normalizer_all, y_test_values_all, symbols