import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })  
    embedding = r.json()["embeddings"]
    return embedding


jsons = os.listdir("jsons")
chunk_id = 0
my_dict = []
for json_file in jsons:
    with open(f"jsons/{json_file}", encoding='utf-8') as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")
    texts = [c["text"] for c in content["chunks"]]   # Creating batches 

    embeddings = []

    for i in range(0, len(texts), 200):
        batch = texts[i:i+200]
        embeddings.extend(create_embedding(batch))


    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk['embeddings'] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)

df = pd.DataFrame.from_records(my_dict)
joblib.dump(df, "embeddings.joblib")