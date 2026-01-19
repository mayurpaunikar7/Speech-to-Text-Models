import whisper
import time
import psutil
import os
from jiwer import wer

# Load model
model = whisper.load_model("tiny")

# Memory after model load
process = psutil.Process(os.getpid())
mem_after_load = process.memory_info().rss / (1024 * 1024)

# Inference timing
start = time.time()
result = model.transcribe("audio1.mp3")
end = time.time()

# Memory after inference
mem_after_infer = process.memory_info().rss / (1024 * 1024)

# Original Lyrics
with open("audio1.txt", "r") as f:
    reference = f.read()

# Metrics
inference_time = end - start
error = wer(reference, result["text"])

print("Transcription:", result["text"])
print(f"Inference Time: {inference_time:.2f} sec")
print(f"WER: {error:.3f}")
print(f"Memory (after load): {mem_after_load:.2f} MB")
print(f"Memory (after inference): {mem_after_infer:.2f} MB")






