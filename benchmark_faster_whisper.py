import time
import psutil
from faster_whisper import WhisperModel
from jiwer import wer

# -------- CONFIG --------
AUDIO_FILE = "audio1.mp3"
GROUND_TRUTH_FILE = "audio1.txt"
MODEL_SIZE = "tiny"   
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  
# ------------------------

process = psutil.Process()

def mem_mb():
    return process.memory_info().rss / (1024 * 1024)

# Original Script
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

print("Loading Faster-Whisper model...")
start_load = time.time()
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
load_time = time.time() - start_load

mem_after_load = mem_mb()

print("Running inference...")
start_inf = time.time()
segments, info = model.transcribe(AUDIO_FILE)
inference_time = time.time() - start_inf

mem_after_inf = mem_mb()

predicted_text = " ".join(segment.text for segment in segments)

# Calculate WER
error = wer(ground_truth.lower(), predicted_text.lower())


print("\n====== Faster-Whisper Results ======")
print(f"Model: faster-whisper-{MODEL_SIZE}")
print(f"Load Time: {load_time:.2f} sec")
print(f"Inference Time: {inference_time:.2f} sec")
print(f"WER: {error:.3f}")
print(f"Memory after load: {mem_after_load:.2f} MB")
print(f"Memory after inference: {mem_after_inf:.2f} MB")
