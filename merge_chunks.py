import os
import math
import json

n = 5

for file_name in os.listdir("jsons"):
    if file_name.endswith(".json"):
        file_path = os.path.join("jsons", file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        new_chunks = []
        num_of_chunks = len(data["chunks"])
        num_of_groups = math.ceil(num_of_chunks/n)

        for i in range(num_of_groups):
            start_idx = i*n
            end_idx = min((i+1)*n, num_of_chunks)

            chunk_group = data["chunks"][start_idx: end_idx]

            new_chunks.append({
                "name" : data["chunks"][0]["name"],
                "video no." : data["chunks"][0]["video no."],
                "start" : chunk_group[0]["start"],
                "end" : chunk_group[-1]["end"],
                "text" : " ".join(c['text'] for c in chunk_group)
            })

        #Saving the file
        os.makedirs("newJsons", exist_ok=True)
        with open(os.path.join("newJsons", file_name), "w", encoding="utf-8") as json_file:
            json.dump({
                "chunks" : new_chunks,
                "text" : data["text"]
            }, json_file, indent=4)