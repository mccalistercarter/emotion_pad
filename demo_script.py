import time
from features.text_features import text_to_pad
from features.audio_features import audio_to_pad
from features.video_features import video_to_pad
from fusion import fuse_pad

turn = {
    "text": "I’m really excited about this project!",
    "audio_raw": "data/example.wav",
    "video_raw": "data/example.mp4"
}

start = time.time()

text_pad = text_to_pad(turn["text"])
audio_pad = audio_to_pad(turn["audio_raw"])
video_pad = video_to_pad(turn["video_raw"])

final_pad = fuse_pad([text_pad, audio_pad, video_pad])

end = time.time()

print("PAD Output:", final_pad)
print("Latency:", round((end - start) * 1000), "ms")