import time
import psutil
import os
from transformers import pipeline
from jiwer import wer

# CONFIG --------
AUDIO_FILE = "audio1.mp3"
GROUND_TRUTH_FILE = "audio1.txt"
MODEL_ID = "distil-whisper/distil-small.en"
DEVICE = "cpu"
# ------------------------

process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / (1024 * 1024)

# Load ground truth
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

print("Loading Distil-Whisper model...")
start_load = time.time()

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    device=-1,           # CPU
    torch_dtype="float32"
)

load_time = time.time() - start_load
mem_after_load = mem_mb()

print("Running inference...")
start_inf = time.time()
result = pipe(AUDIO_FILE, return_timestamps=True)
inference_time = time.time() - start_inf

mem_after_inf = mem_mb()

predicted_text = result["text"]

error = wer(
    ground_truth.lower(),
    predicted_text.lower()
)

print("\n====== Distil-Whisper Results ======")
print(f"Model: {MODEL_ID}")
print(f"Load Time: {load_time:.2f} sec")
print(f"Inference Time: {inference_time:.2f} sec")
print(f"WER: {error:.3f}")
print(f"Memory after load: {mem_after_load:.2f} MB")
print(f"Memory after inference: {mem_after_inf:.2f} MB")
