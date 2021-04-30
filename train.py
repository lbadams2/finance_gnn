import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from params import *
import load_data

def train_gnn(embeddings, neighbors, ground_truth):
    num_nodes = embeddings.shape[0]
    node_features = tf.Variable(tf.random_normal_initializer()(shape=[num_nodes, node_feature_dim], dtype=tf.float32))
    relation_embedding = tf.Variable(tf.random_normal_initializer()(shape=[relation_embedding_dim], dtype=tf.float32))
    concat_dim = node_feature_dim + node_feature_dim + relation_embedding_dim
    weight_matrix = tf.Variable(tf.random_normal_initializer()(shape=[num_nodes, concat_dim, weight_out_dim], dtype=tf.float32))
    bias = tf.zeros([num_nodes, weight_out_dim], dtype=tf.float32)

    final_prediction_matrices = tf.Variable(tf.random_normal_initializer()(shape=[num_nodes, weight_out_dim, 1], dtype=tf.float32))
    final_biases = tf.zeros([num_nodes, 1], dtype=tf.float32)
    # embedding is input, should be tensor of shape [num_nodes, embedding_dim]

    trainable_variables = [node_features, relation_embedding, weight_matrix, bias, final_prediction_matrices, final_biases]
    # weight matrix and final_prediction_matrices gradients aren't calculating
    del trainable_variables[3]
    del trainable_variables[4]
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    num_iterations = ground_truth.shape[0]
    # entire windows were embedded
    for n in range(num_iterations): # timesteps
        if n >= 5:
            break
        print('training time step', n)
        with tf.GradientTape() as tape:
            predictions = []
            for i in range(num_nodes):
                current_node_embedding_input = tf.gather(embeddings, i)
                current_node_embedding_input = tf.gather(current_node_embedding_input, n)
                current_node_feature = tf.gather(node_features, i)
                neighbor_indices = neighbors[i]
                stacked_neighbors = []
                stacked_neighbors_input_emb = []
                for j in neighbor_indices:
                    emb_neighbor = tf.gather(embeddings, i)
                    emb_neighbor = tf.gather(emb_neighbor, n)
                    stacked_neighbors_input_emb.append(emb_neighbor)
                    node_feature_neighbor = tf.gather(node_features, j) # j should be tensor
                    
                    concat_tensor = tf.concat([current_node_feature, node_feature_neighbor, relation_embedding], axis=0)
                    node_weight_matrix = tf.gather(weight_matrix, i)
                    node_bias = tf.gather(bias, i)
                    #product = tf.linalg.matmul(tf.expand_dims(concat_tensor,0), node_weight_matrix)
                    product = tf.tensordot(concat_tensor, node_weight_matrix, 1)
                    product = tf.squeeze(product)
                    transformed = tf.math.add(product, node_bias)
                    
                    exponential_transform = tf.math.exp(transformed)
                    stacked_neighbors.append(exponential_transform)

                stacked_neighbors = tf.stack(stacked_neighbors)
                stacked_neighbors_input_emb = tf.stack(stacked_neighbors_input_emb)
                stacked_neighbors_transformed = tf.math.divide(stacked_neighbors, tf.math.reduce_sum(stacked_neighbors, axis=0))
                node_representation = tf.math.multiply(stacked_neighbors_transformed, stacked_neighbors_input_emb)
                node_representation_reduced = tf.math.reduce_sum(node_representation, axis=0)
                final_node_representation = tf.math.add(node_representation_reduced, current_node_embedding_input)
                
                final_pred_matrix = tf.gather(final_prediction_matrices, i)
                final_bias = tf.gather(final_biases, i)
                final_product = tf.linalg.matmul(tf.expand_dims(final_node_representation,0), final_pred_matrix)
                prediction = tf.math.add(final_product, final_bias)
                predictions.append(prediction)

            predictions = tf.stack(predictions)
            current_gt = tf.gather(ground_truth, n)
            diff = tf.math.subtract(predictions, current_gt)
            loss = tf.math.square(diff)
            reduced_loss = tf.math.reduce_sum(loss)
        
        grads = tape.gradient(reduced_loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/node_features.npy', 'wb') as f:
        np.save(f, node_features.numpy())
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/relation_embedding.npy', 'wb') as f:
        np.save(f, relation_embedding.numpy())
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/weight_matrix.npy', 'wb') as f:
        np.save(f, weight_matrix.numpy())
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/bias.npy', 'wb') as f:
        np.save(f, bias.numpy())
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/final_prediction_matrices.npy', 'wb') as f:
        np.save(f, final_prediction_matrices.numpy())
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/final_biases.npy', 'wb') as f:
        np.save(f, final_biases.numpy())

def get_symbols(members):
    symbols = []
    dim = members.shape[0]
    for i in range(dim):
        symbol = members.iloc[[i]]['tickerLabel'].to_numpy()[0]
        symbols.append(symbol)
    return symbols

def predict(x_test_embeddings, neighbors, y_test_values, y_test_scalers):
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/node_features.npy', 'rb') as f:
        node_features = np.load(f)
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/relation_embedding.npy', 'rb') as f:
        relation_embedding = np.load(f)
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/weight_matrix.npy', 'rb') as f:
        weight_matrix = np.load(f)
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/bias.npy', 'rb') as f:
        bias = np.load(f)
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/final_prediction_matrices.npy', 'rb') as f:
        final_prediction_matrices = np.load(f)
    with open('/Users/liam_adams/my_repos/finance_gnn/gnn/final_biases.npy', 'rb') as f:
        final_biases = np.load(f)
    
    timesteps = y_test_values.shape[0]
    num_nodes = x_test_embeddings.shape[0]
    
    # entire windows were embedded
    all_predictions = []
    for n in range(timesteps):
        print('testing time step', n)
        predictions = []
        for i in range(num_nodes):
            current_node_embedding_input = x_test_embeddings[i, n]
            current_node_feature = node_features[i]
            neighbor_indices = neighbors[i]
            stacked_neighbors = []
            stacked_neighbors_input_emb = []
            for j in neighbor_indices:
                emb_neighbor = x_test_embeddings[i, n]
                stacked_neighbors_input_emb.append(emb_neighbor)
                node_feature_neighbor = node_features[j]
                
                concat_arr = np.concat([current_node_feature, node_feature_neighbor, relation_embedding], axis=0)
                node_weight_matrix = weight_matrix[i]
                node_bias = bias[i]
                product = np.dot(concat_tensor, node_weight_matrix)
                product = tf.squeeze(product)
                transformed = np.add(product, node_bias)
                
                exponential_transform = np.exp(transformed)
                stacked_neighbors.append(exponential_transform)

            stacked_neighbors = np.stack(stacked_neighbors)
            stacked_neighbors_input_emb = np.stack(stacked_neighbors_input_emb)
            stacked_neighbors_transformed = stacked_neighbors / np.reduce_sum(stacked_neighbors, axis=0)
            node_representation = stacked_neighbors_transformed * stacked_neighbors_input_emb
            node_representation_reduced = np.reduce_sum(node_representation, axis=0)
            final_node_representation = node_representation_reduced + current_node_embedding_input
            
            final_pred_matrix = final_prediction_matrices[i]
            final_bias = final_biases[i]
            final_product = np.dot(final_node_representation, final_pred_matrix)
            prediction = final_product + final_bias
            real_prediction = y_test_scaler[i].inverse_transform(prediction)   
            predictions.append(real_prediction)

        #y_test_predicted = y_test_scaler.inverse_transform(predictions)
        all_predictions.append(predictions)

    all_predictions_arr = np.asarray(all_predictions)
    real_predictions = all_predictions_arr - y_test_values
    all_profit_val = real_predictions.sum()


if __name__ == '__main__':
    x_train_windows, x_test_windows, y_train_norm, y_train_values, y_train_normalizer, \
                    ti_train_norm, ti_test_norm, y_test_norm, y_test_normalizer_all, y_test_values, symbols = load_data.get_regression_training_data()
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')    
    member_graph = load_data.load_graph()
    
    members, member_graph, y_train_norm = load_data.delete_nodes(members, member_graph, y_train_norm)
    symbols = get_symbols(members)
    neighbor_map = load_data.get_neighbors(member_graph, symbols)
    all_embeddings = load_data.load_embeddings(members)
    embeddings_tensor = tf.convert_to_tensor(all_embeddings, dtype=tf.float32)

    dims = y_train_norm.shape
    y_train_norm = y_train_norm.reshape((dims[1], dims[0], dims[2]))
    y_train_tensor = tf.convert_to_tensor(y_train_norm, dtype=tf.float32)
    
    train_gnn(embeddings_tensor, neighbor_map, y_train_tensor)
    
    '''
    x_test_embeddings = load_data.load_embeddings(members, False)
    predict(x_test_embeddings, neighbor_map, y_test_values, y_test_normalizer_all)
    '''