import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import json

def embedding_creater(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"] 
    return embedding

data = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = embedding_creater([incoming_query])[0] 

similarities = cosine_similarity(np.vstack(data['embedding']), [question_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
Data = data.loc[max_indx] 
for index, item in Data.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])