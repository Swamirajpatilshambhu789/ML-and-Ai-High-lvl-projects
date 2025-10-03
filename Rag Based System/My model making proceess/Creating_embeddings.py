import numpy as np
import requests
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def embedding_creater(text_list):
    req = requests.post('http://localhost:11434/api/embed', json={
        "model":'bge-m3',
        "input": text_list,
    })

    embeddings = req.json()['embeddings']
    return embeddings

chunks_list = []
chunk_id = 0

jsons = os.listdir("jsons")
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")
    embeddings = embedding_creater([c['text'] for c in content['chunks']])

    for i,chunk in enumerate(content['chunks']):
        chunk['id'] = chunk_id
        chunk_id = chunk_id + 1
        chunk['embedding'] = embeddings[i]
        chunks_list.append(chunk)

Data = pd.DataFrame.from_records(chunks_list)
joblib.dump(Data, 'embeddings.joblib')