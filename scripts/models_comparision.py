import time
import psutil
import os
from jiwer import wer


AUDIO_FILE = "audio1.mp3"
GROUND_TRUTH_FILE = "audio1.txt"
DEVICE = "cpu" 
# ------------------------

process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / (1024 * 1024)

# Original lyrics
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

results = []

# Distil-Whisper ----------------
from transformers import pipeline

print("Benchmarking Distil-Whisper...")
start_load = time.time()
distil_pipe = pipeline(
    task="automatic-speech-recognition",
    model="distil-whisper/distil-small.en",
    device=-1,  # CPU
    torch_dtype="float32"
)
load_time = time.time() - start_load
mem_after_load = mem_mb()

start_inf = time.time()
distil_out = distil_pipe(AUDIO_FILE, return_timestamps=True)
inference_time = time.time() - start_inf
mem_after_inf = mem_mb()

pred_text = distil_out["text"]
error = wer(ground_truth.lower(), pred_text.lower())

results.append({
    "Model": "Distil-Whisper",
    "Load Time (s)": round(load_time, 2),
    "Inference Time (s)": round(inference_time, 2),
    "WER": round(error, 3),
    "Memory After Load (MB)": round(mem_after_load, 2),
    "Memory After Inference (MB)": round(mem_after_inf, 2)
})

# Faster-Whisper ----------------
from faster_whisper import WhisperModel

print("Benchmarking Faster-Whisper...")
start_load = time.time()
faster_model = WhisperModel("tiny", device=DEVICE, compute_type="int8")
load_time = time.time() - start_load
mem_after_load = mem_mb()

start_inf = time.time()
segments, info = faster_model.transcribe(AUDIO_FILE)
inference_time = time.time() - start_inf
mem_after_inf = mem_mb()

pred_text = " ".join(segment.text for segment in segments)
error = wer(ground_truth.lower(), pred_text.lower())

results.append({
    "Model": "Faster-Whisper",
    "Load Time (s)": round(load_time, 2),
    "Inference Time (s)": round(inference_time, 2),
    "WER": round(error, 3),
    "Memory After Load (MB)": round(mem_after_load, 2),
    "Memory After Inference (MB)": round(mem_after_inf, 2)
})

# Whisper (tiny) ----------------
import whisper

print("Benchmarking Whisper-tiny...")
start_load = time.time()
whisper_model = whisper.load_model("tiny")
load_time = time.time() - start_load
mem_after_load = mem_mb()

start_inf = time.time()
result = whisper_model.transcribe(AUDIO_FILE)
inference_time = time.time() - start_inf
mem_after_inf = mem_mb()

pred_text = result["text"]
error = wer(ground_truth.lower(), pred_text.lower())

results.append({
    "Model": "Whisper-tiny",
    "Load Time (s)": round(load_time, 2),
    "Inference Time (s)": round(inference_time, 2),
    "WER": round(error, 3),
    "Memory After Load (MB)": round(mem_after_load, 2),
    "Memory After Inference (MB)": round(mem_after_inf, 2)
})

# Print Comparison Table ----------------
print("\n====== ASR Model Benchmark Comparison ======")
header = ["Model", "Load Time (s)", "Inference Time (s)", "WER",
          "Memory After Load (MB)", "Memory After Inference (MB)"]
row_format = "{:<20} {:<15} {:<17} {:<8} {:<22} {:<25}"
print(row_format.format(*header))
print("-"*110)
for r in results:
    print(row_format.format(r["Model"], r["Load Time (s)"], r["Inference Time (s)"],
                            r["WER"], r["Memory After Load (MB)"], r["Memory After Inference (MB)"]))
