import numpy as np
import pandas as pd
import pandas_datareader.data as web

sp_500_sparql = """SELECT ?item ?itemLabel ?exchangeLabel ?tickerLabel ?industryLabel ?ownedbyLabel ?productLabel ?memberLabel ?boardmemberLabel
WHERE 
{
  ?item wdt:P361 wd:Q242345 ; p:P414 [pq:P249 ?ticker; ps:P414 ?exchange ]; p:P452 [ps:P452 ?industry]; 
        p:P127 [ps:P127 ?ownedby]; p:P1056 [ps:P1056 ?product]; p:P463 [ps:P463 ?member]; p:P3320 [ps:P3320 ?boardmember] .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}"""

def get_sp_500_symbols():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    wiki_sp500 = table[0]
    wiki_sp500.to_csv('S&P500-Info.csv')
    sp_500_symbols = wiki_sp500['Symbol']
    return sp_500_symbols


# Blackrock owns Apple, Blackrock owns google, second order relation
# Todd is a board member at Google
# Todd founded apple
# Google and apple are connected in the graph consisting of (Board member, founded by) relations
# (industry, material or product produced), (industry, manufacturer), member of, board member
# create adjacency matrix
def load_graphs():
    with open('data/industry_graph.npy', 'rb') as f:
        industry_graph = np.load(f)
    with open('data/board_graph.npy', 'rb') as f:
        board_graph = np.load(f)
    with open('data/member_graph.npy', 'rb') as f:
        member_graph = np.load(f)
    with open('data/ownedby_graph.npy', 'rb') as f:
        ownedby_graph = np.load(f)
    

if __name__ == '__main__':
    all_prices = pd.read_csv('data/historical_prices.csv')
    prices = all_prices['Adj Close']
