import whisper
import json
import os

audios = os.listdir("audios")
model = whisper.load_model("large-v2")

for audio in audios:
    number = audio.split("_")[0].strip()
    title = audio.split("_")[1][:-4].strip()
    result = model.transcribe(audio= f"audios/{audio}",
                              language="hi",
                              task="translate",
                              word_timestamps=False)
    chunks = []
    for segment in result["segments"]:
        chunks.append({"name" : title, "video no." : number, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})

    chunks_with_script = {"chunks" : chunks, "text" : result["text"]}

    with open(f"jsons/{number} {title}.json", "w") as f:
        json.dump(chunks_with_script, f)