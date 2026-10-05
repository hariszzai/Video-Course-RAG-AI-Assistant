import os
import subprocess

files = os.listdir("videos")
for file in files:
    video_number = file.split(" - CodeWithHarry")[0].split("#")[1]
    video_name = file.split("  Sigma")[0]
    print(video_number, video_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{video_number}_{video_name}.mp3"])