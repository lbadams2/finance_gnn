import numpy as np
import pandas as pd
import load_data

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
                
                concat_arr = np.concatenate([current_node_feature, node_feature_neighbor, relation_embedding], axis=0)
                node_weight_matrix = weight_matrix[i]
                node_bias = bias[i]
                product = np.dot(concat_arr, node_weight_matrix)
                #product = tf.squeeze(product)
                transformed = np.add(product, node_bias)
                
                exponential_transform = np.exp(transformed)
                stacked_neighbors.append(exponential_transform)

            stacked_neighbors = np.stack(stacked_neighbors)
            stacked_neighbors_input_emb = np.stack(stacked_neighbors_input_emb)
            stacked_neighbors_transformed = stacked_neighbors / np.sum(stacked_neighbors, axis=0)
            node_representation = stacked_neighbors_transformed * stacked_neighbors_input_emb
            node_representation_reduced = np.sum(node_representation, axis=0)
            final_node_representation = node_representation_reduced + current_node_embedding_input
            
            final_pred_matrix = final_prediction_matrices[i]
            final_bias = final_biases[i]
            final_product = np.dot(final_node_representation, final_pred_matrix)
            prediction = final_product + final_bias
            real_prediction = y_test_scalers[i].inverse_transform(np.expand_dims(prediction, axis=0))
            predictions.append(real_prediction)

        #y_test_predicted = y_test_scaler.inverse_transform(predictions)
        all_predictions.append(predictions)

    all_predictions_arr = np.asarray(all_predictions)
    real_predictions = all_predictions_arr - y_test_values
    all_profit_val = real_predictions.sum()

# percent return of each next day price
def next_day_return(x_test_windows, y_test_norm):
    print('')

if __name__ == '__main__':
    x_train_windows, x_test_windows, y_train_norm, y_train_values, y_train_normalizer, \
                    ti_train_norm, ti_test_norm, y_test_norm, y_test_normalizer_all, y_test_values, symbols = load_data.get_regression_training_data()
    members = pd.read_csv('/Users/liam_adams/my_repos/finance_gnn/data/members.csv')    
    member_graph = load_data.load_graph()
    
    members, member_graph, y_test_norm, x_test_windows = load_data.delete_nodes(members, member_graph, y_test_norm, x_test_windows)
    y_test_norm = y_test_norm.squeeze()
    dims = y_test_norm.shape
    y_test_norm = y_test_norm.reshape((dims[1], dims[0]))

    symbols = load_data.get_symbols(members)
    neighbor_map = load_data.get_neighbors(member_graph, symbols)
    
    x_test_embeddings = load_data.load_embeddings(members, False)
    next_day_return(x_test_windows, y_test_norm)
    predict(x_test_embeddings, neighbor_map, y_test_values, y_test_normalizer_all)