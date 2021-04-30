embedding_dim = 32
lookback_days = 50
num_training_features = 5
node_feature_dim = 20
relation_embedding_dim = 12
weight_out_dim = 32

sp_500_sparql = """SELECT ?item ?itemLabel ?exchangeLabel ?tickerLabel ?industryLabel ?ownedbyLabel ?productLabel ?memberLabel ?boardmemberLabel
WHERE 
{
  ?item wdt:P361 wd:Q242345 ; p:P414 [pq:P249 ?ticker; ps:P414 ?exchange ]; p:P452 [ps:P452 ?industry]; 
        p:P127 [ps:P127 ?ownedby]; p:P1056 [ps:P1056 ?product]; p:P463 [ps:P463 ?member]; p:P3320 [ps:P3320 ?boardmember] .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}"""