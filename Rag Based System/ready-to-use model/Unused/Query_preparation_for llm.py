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
prompt_for_llm = f''' I am teaching Web dev using Sigma Web devlopment course
Here are video subtitle chunks containg video title, video number , start time in seconds, end time in seconds
the text at that time:

{Data[['title', 'number', 'start', 'end', 'text']].to_json()}
----------------------------------------------------------------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video.
If the user askes unrelated question tell him you can only answer question from the course
'''

with open("prompt.txt", "w") as f :
    f.write(prompt_for_llm)
# for index, item in Data.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])