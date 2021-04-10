import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam

node_feature_dim = 64
lookback_days = 50
num_training_features = 5

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
    #norm_prices = normalize_prices(np_prices)
    return np_prices

def get_neighbors(graph, symbol_index):
    row = graph[symbol_index]
    col = graph[ : , symbol_index]
    row_inds = np.where(row == 1)[0]
    #col_inds = np.where(col == 1)[0]
    #all_inds = np.concatenate((row_inds, col_inds))
    #return all_inds
    return row_inds

def get_regression_training_data(member_graph, members):    
    row_sums = np.sum(member_graph, axis=1).tolist()
    #col_sums = member_graph.sum(axis=0)
    member_graph = np.maximum(member_graph, member_graph.transpose()) # make matrix symmetric
    node_map = {}
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

    dim = member_graph.shape[0]
    nodes_to_delete = []
    node_count = 0
    for i in range(dim):
        if row_sums[i] < 1:
            nodes_to_delete.append(i)
        else:
            ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0]
            node_map[node_count] = ticker
            node_count += 1

    member_graph = np.delete(member_graph, nodes_to_delete, axis=0) # delete nodes with no edges
    member_graph = np.delete(member_graph, nodes_to_delete, axis=1) # delete nodes with no edges
    dim = member_graph.shape[0]
    for i in range(dim):
        ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0] # stocks in alphabetical order in member_graph and members
        prices = get_prices(ticker)
        #node_map[i] = ticker
        neighbors = get_neighbors(member_graph, i)
        neighbor_map[i] = neighbors
        test_split = 0.9 # the percent of data to be used for testing
        n = int(prices.shape[0] * test_split)
        x_train = prices[:n]
        x_train_norm = normalize_prices(x_train)
        x_test = prices[n:]
        x_test_norm = normalize_prices(x_test)
        # changes to slices will reflect in original so use copy, i + lookback_days not included
        x_train_windows = np.array([x_train_norm[i  : i + lookback_days].copy() for i in range(len(x_train_norm) - lookback_days)])
        x_train_windows_all.append(x_train_windows)
        x_test_windows = np.array([x_test_norm[i  : i + lookback_days].copy() for i in range(len(x_test_norm) - lookback_days)])
        x_test_windows_all.append(x_test_windows)
        
        # value of next day close for each x window, predicting next day close
        next_day_close_values_norm = np.array([x_train_norm[:,0][i + lookback_days].copy() for i in range(len(x_train_norm) - lookback_days)])
        next_day_close_values_norm = np.expand_dims(next_day_close_values_norm, -1) # make 2D
        y_train_norm = next_day_close_values_norm
        y_train_norm_all.append(y_train_norm)

        next_day_close_values = np.array([x_train[:,0][i + lookback_days] for i in range(len(x_train) - lookback_days)])
        # expand_dims?
        y_train_values = next_day_close_values
        y_train_values_all.append(y_train_values)

        y_train_normalizer = MinMaxScaler()
        y_train_normalizer.fit(np.expand_dims( y_train_values, -1 )) # allows us to un normalize at the end
        y_train_normalizer_all.append(y_train_normalizer)

        test_next_day_close_values_norm = np.array([x_test_norm[:,0][i + lookback_days].copy() for i in range(len(x_test_norm) - lookback_days)])
        test_next_day_close_values_norm = np.expand_dims(test_next_day_close_values_norm, -1) # make 2D
        y_test_norm = test_next_day_close_values_norm
        y_test_norm_all.append(y_test_norm)

        test_next_day_close_values = np.array([x_test[:,0][i + lookback_days] for i in range(len(x_test) - lookback_days)])
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
    y_train_normalizer_all = np.stack(y_train_normalizer_all)
    ti_train_norm_all = np.stack(ti_train_norm_all)
    ti_test_norm_all = np.stack(ti_test_norm_all)
    y_test_norm_all = np.stack(y_test_norm_all)
    y_test_values_all = np.stack(y_test_values_all)

    assert x_train_windows_all.shape[0] == y_train_norm_all.shape[0] == ti_train_norm_all.shape[0]
    return x_train_windows_all, x_test_windows_all, y_train_norm_all, y_train_values_all, y_train_normalizer_all, \
                    ti_train_norm_all, ti_test_norm_all, y_test_norm_all, y_test_values_all, node_map, neighbor_map

def create_gnn(model, neighbors, symbol):
    # for each neighbor concatenate its representation with current node's representation and relation embedding
    # concatenate these 3 vectors and feed to attention layer that has weight and bias as learnable params
    # pass each concatenated vector for each neighbor through the attention layer and sum them
    # this summed vector is used as input to the relation attention layer which has learnable weight and bias
    # the relation attention layer sums the all of these relation vectors for the company
    # the output of this is the final node embedding, this is added to the embedding output by the LSTM layer
    # now add output layers for prediction and calculate loss
    pass

def create_model_regression(ti_shape, num_nodes, neighbor_map):
    lstm_inputs = []
    ti_inputs = []
    outputs = []
    for i in range(num_nodes):
        lstm_input = Input(shape=(lookback_days, num_training_features), name='lstm_input_' + str(i))
        dense_input = Input(shape=(ti_shape,), name='tech_input_' + str(i))
        
        # the first branch operates on the first input, each company has its own LSTM
        x = LSTM(lookback_days, name='lstm_' + str(i))(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_' + str(i))(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x)
        
        # the second branch opreates on the second input
        y = Dense(20, name='tech_dense_' + str(i))(dense_input)
        y = Activation("relu", name='tech_relu_' + str(i))(y)
        y = Dropout(0.2, name='tech_dropout_' + str(i))(y)
        technical_indicators_branch = Model(inputs=dense_input, outputs=y)

        # can add third branch for textual data
        
        # combine the output of the two branches
        combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate_' + str(i))
        
        z = Dense(node_feature_dim, activation="relu", name='dense_pooling_' + str(i))(combined) # save this for the node embedding
        # concatenate neighbors representation with current node representation
        lstm_inputs.append(lstm_branch.input)
        ti_inputs.append(technical_indicators_branch.input)
        outputs.append(z)
    #model = Model(inputs=[lstm_inputs, ti_inputs], outputs=outputs)

    gnn_outputs = []
    # need to make graph matrix symmetrical for this
    for i in range(num_nodes):
        neighbors = neighbor_map[i]
        #neighbors = tf.convert_to_tensor(neighbors)
        current_node = outputs[i]
        neighbor_tensors = []
        z = Dense(node_feature_dim, activation="relu") # don't use activation here if predicting categories
        neighbors_stacked = []
        for n in neighbors:
            # create dense layer which has weight and bias
            # if you have multiple graphs, concatenate an overall graph embedding vector here as well
            neighbors_stacked.append(outputs[n])
            t = tf.concat([current_node, outputs[n]], -1)
            t = z(t) # reduce dimensions back down to node_feature_dim
            #t = tf.math.exp(t)
            neighbor_tensors.append(t)        
        neighbor_tensors = tf.stack(neighbor_tensors)
        #tensor_sum = tf.reduce_sum(neighbor_tensors)
        #neighbor_tensors = tf.nn.softmax(neighbor_tensors) # softmax used for categorical prediction
        #neighbor_tensors = tf.math.divide(neighbor_tensors, tensor_sum)
        neighbors_stacked = tf.convert_to_tensor(neighbors_stacked)
        product = neighbor_tensors * neighbors_stacked
        tensor_sum = tf.reduce_sum(product)
        gnn_outputs.append(tensor_sum)
        # can add relation attention layer here if multiple graphs
        # tensor_sum is equation 3.3 in HATS
    gnn_outputs = tf.expand_dims(tf.convert_to_tensor(gnn_outputs), -1)
    out = Dense(num_nodes, activation="relu")(gnn_outputs)
    model = Model(inputs=[lstm_inputs, ti_inputs], outputs=out)

    adam = Adam(lr=0.0005)

    model.compile(optimizer=adam,
                loss='mse')
    #model.summary()
    return model

# create f dimensional vector for each stock
# need to split training and test data before normalizing, use fit_transform on training data, transform on test data
# minmaxscaler remembers mean and variance from fit_transform to scale test data accordingly, fit_transform first
# also need to handle NANs if any
def train(member_graph, members):
    #test_split = 0.9 # the percent of data to be used for testing
    #n = int(x_windows.shape[0] * test_split)
    # splitting the dataset up into train and test sets
    #x_train = x_windows[:n]
    #ti_train = ti_norm[:n]
    #y_train = next_day_close_values_norm[:n]

    #x_test = x_windows[n:]
    #ti_test = ti_norm[n:]
    #y_test = next_day_close_values_norm[n:]

    #unscaled_y_test = next_day_close_values[n:]
    x_train_windows, x_test_windows, y_train_norm, y_train_values, y_train_normalizer, \
                    ti_train_norm, ti_test_norm, y_test_norm, y_test_values, node_map, neighbor_map = get_regression_training_data(member_graph, members)
    model = create_model_regression(ti_train_norm.shape[2], ti_train_norm.shape[0], neighbor_map)

    # x_train_windows is node_num x num_windows x lookback_days x input_features np array

    # make sure validation data is newer than training data
    model.fit(x=[x_train_windows, ti_train_norm], y=y_train_norm, batch_size=32, epochs=10, shuffle=False, validation_split=0.1)
    evaluation = model.evaluate([x_test_windows, ti_test_norm], y_test_norm)
    #model.save_weights('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + symbol)
    #model.save('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + symbol)
    #test = tf.keras.models.load_model('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + symbol)
    #test.layers[9].weights[0].numpy()


if __name__ == '__main__':
    member_graph = load_graph()
    dim = member_graph.shape[0] # comes from members.csv, symbols in alhpabetical order
    print('nodes in graph', dim)
    row_sums = np.sum(member_graph, axis=1).tolist()
    col_sums = member_graph.sum(axis=0)
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')
    train(member_graph, members)
    # if ith row or column has a 1, node i is in the graph
    