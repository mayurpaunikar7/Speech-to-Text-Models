# Speech-to-Text-Models
Comparative Study of Speech-to-Text Models for Noisy Real-World Audio


# 🎙️ ASR Model Benchmarking – Noisy Real-World Audio

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)

---

## 📌 Project Overview
Benchmarking **state-of-the-art ASR models** to identify the **best performer** for noisy customer support audio with multiple accents.

---

## 🎯 Objective
- Compare **accuracy**, **inference speed**, **memory usage**, and **deployment readiness**
- Recommend the **most suitable model for production**

---

## 🧠 Models Evaluated

| Model | Key Features |
|-------|--------------|
| **Whisper-Tiny** | Robust to noise, slower CPU inference |
| **Distil-Whisper** | Fast, memory-efficient, distilled |
| **Faster-Whisper** | Fastest CPU inference, low memory, production-ready |

---

## 📂 Dataset
- Audio format: `.mp3`  
- Ground truth: `.txt` transcripts  
- Scenario: Single-speaker, noisy real-world audio  

---

## 📊 Evaluation Metrics
- **Word Error Rate (WER)** – transcription accuracy  
- **Inference Time** – speed on CPU  
- **Memory Usage** – after load and after inference  

---

## 🏁 Benchmark Results

| Model           | WER   | Inference Time | Memory Usage | Recommendation |
|----------------|-------|----------------|--------------|----------------|
| Distil-Whisper | High  | Moderate       | Low          | Moderate       |
| Whisper-Tiny   | Medium| Slow           | High         | Low            |
| Faster-Whisper | Low   | Fast           | Low          | ✅ Best        |

> **Key takeaway:** **Faster-Whisper** balances accuracy, speed, and memory for noisy audio.

---

## ⚡ Usage (Demo)
```bash
# Run Distil-Whisper
python benchmark_distil_whisper.py

# Run Faster-Whisper
python benchmark_faster_whisper.py

# Run Whisper-Tiny
python STT.py

