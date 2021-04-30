import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, concatenate, Input
from tensorflow.keras.optimizers import Adam
from load_data import get_regression_training_data
import params
import matplotlib.pyplot as plt

'''
input to lstm is matrix for an individual stock, there are num_lookback_days columns in the matrix
columns are (open, high, low, close, volume)
the last hidden state of the lstm is the embedding of the stock used in the graph

use GNN to update all embeddings
then pass to FC layer to get (up, down, neutral)

need relational embedding vector for overall graph
node embeddings can be use to make the strenth of the relational embedding increase/decrease over time

to predict numerical price use FC layer outputting 1 and mse for loss

first train an lstm for each stock independently and use the last hidden layer of the lstm output as stock embedding
save these embeddings to file
then load them and concatenate them into tensor, select columns from the tensor depending on current node and its neighbors
'''

def train(x_train, y_train, ticker, save=False):
    print('training ', ticker)
    inputs1=Input(shape=(params.lookback_days, params.num_training_features))
    # embedding dim is size of hidden state
    lstm1, states_h, states_c =LSTM(params.embedding_dim,dropout=0.3,recurrent_dropout=0.2, return_state=True)(inputs1)
    z = Dense(units=1)(lstm1)
    model = Model(inputs1,z)
    
    state_getting_model = Model(inputs1, [lstm1, states_h, states_c]) 
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(x_train, y_train, epochs=30)
    
    # this will be windows x embedding dim
    _, hidden_state, _ = state_getting_model.predict(x_train)

    if save:
        with open('/Users/liam_adams/my_repos/finance_gnn/embeddings/' + ticker + '.npy', 'wb') as f:
            np.save(f, hidden_state)
    
    '''
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('embedding loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    '''
    print('\n\n\n')


if __name__ == '__main__':
    x_train_windows, x_test_windows, y_train_norm, y_train_values, y_train_normalizer, \
                    ti_train_norm, ti_test_norm, y_test_norm, y_test_normalizer_all, y_test_values, symbols = get_regression_training_data()
    state_models = []
    for x, y, ticker in zip(x_train_windows, y_train_norm, symbols):
        train(x, y, ticker)
    #for x, y, ticker in zip(x_test_windows, y_test_norm, symbols):
    #    train(x, y, ticker, True)