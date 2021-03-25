import numpy as np
import pandas as pd

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

def train_node(node_num, ticker, prices):
    pass


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
            # first column of prices is date, stocks in alphabetical order
            prices = historical_prices.iloc[:, i + 1]
            # remove first 2 rows that don't have price data
            prices = prices.iloc[2:]
            train_node(i, ticker, prices)
    