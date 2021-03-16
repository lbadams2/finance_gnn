import numpy as np
import pandas as pd
import pandas_datareader.data as web

sp_500_sparql = """SELECT ?item ?itemLabel ?exchangeLabel ?tickerLabel ?industryLabel
WHERE 
{
  ?item wdt:P361 wd:Q242345 ; p:P414 [pq:P249 ?ticker; ps:P414 ?exchange ]; p:P452 [ps:P452 ?industry] .
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
def create_graph(sp_500_symbols):
    company_code = 'Q783794'
    sp_500_wiki_code = 'Q242345'
    

if __name__ == '__main__':
    all_prices = pd.read_csv('data/historical_prices.csv')
    prices = all_prices['Adj Close']
