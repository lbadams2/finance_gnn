import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam

embedding_dim = 64
lookback_days = 50
num_training_features = 5

historical_prices = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/historical_prices.csv')
hist_symbols = historical_prices.iloc[[0]].to_numpy().squeeze()
sp500 = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/S&P500-Info.csv')
symbols = sp500['Symbol']

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

def load_data(lookback_days):
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')
    dim = members.shape[0]
    all_prices = []
    for i in range(dim):
        ticker = members.iloc[[i]]['tickerLabel'].to_numpy()[0]
        prices = get_prices(ticker)
        all_prices.append(prices)
    x_train = np.stack(all_prices)
    y_train = x_train[:,0]
    y_train = y_train[0::lookback_days]    
    return x_train, y_train

def train(num_companies, lookback_days, input_features, x_train, y_train):
    x_train, y_train = load_data(lookback_days)
    lstm_input = Input(shape=(num_companies, lookback_days, input_features), name='lstm_input_0')
    x = LSTM(embedding_dim, name='lstm_0')(lstm_input)
    z = Dense(1)(x)
    model = Model(inputs=[lstm_input.input], outputs=z)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=5, batch_size=num_companies, shuffle=False)

def test_rnn():
    pass

if __name__ == '__main__':
    train()