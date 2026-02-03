# Speech-to-Text-Models
Comparative Study of Speech-to-Text Models for Noisy Real-World Audio


# 🎙️ ASR Model Benchmarking – Noisy Real-World Audio

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)

---

## 📌 Project Overview
This project benchmarks **state-of-the-art Automatic Speech Recognition (ASR) models** to identify the **best model for noisy, real-world customer support audio**, where:

- Background noise is present  
- Accents may vary  
- Low latency and low memory usage are critical  

The goal is to simulate **production-like constraints** and recommend a **deployment-ready ASR model**.

---

## 🎯 Objective
- Compare ASR models on:
  - **Accuracy** (Word Error Rate)
  - **Inference Speed** (Latency)
  - **Memory Consumption**
- Evaluate **CPU-only inference** (realistic for cost-sensitive deployments)
- Recommend the **best model for production use**

---

## Models Evaluated

| Model | Description |
|------|------------|
| **Whisper-Tiny** | Original OpenAI Whisper model, small size, robust but slower on CPU |
| **Distil-Whisper** | Knowledge-distilled Whisper, faster and more memory-efficient |
| **Faster-Whisper** | Optimized CTranslate2-based Whisper, designed for production ||

---

## Why These Models?

- **Whisper-Tiny** → Baseline OpenAI model  
- **Distil-Whisper** → Reduced size, faster inference  
- **Faster-Whisper** → Optimized backend, quantized inference, production-grade  

This gives a **fair comparison between research vs optimized models**.

---

## 📂 Dataset
- Audio format: `.mp3`  
- Ground truth: `.txt` transcripts  
- Scenario: Single-speaker, noisy real-world audio
- **Why small samples?**
  - Avoid large dataset downloads
  - Focus on **model behavior**, not dataset scale
  - Suitable for **rapid benchmarking & interviews**

---

## 📏 Evaluation Metrics

### 1️ Word Error Rate (WER)
Measures transcription accuracy.

\[
WER = \frac{Substitutions + Deletions + Insertions}{Total\ Words}
\]

Lower WER = better accuracy.

---

### 2️ Inference Time
- Measures how long the model takes to transcribe audio
- Important for **real-time or near-real-time systems**

---

### 3️ Memory Usage
- Measured:
  - After model load
  - After inference
- Important for **edge devices & cost-optimized servers**

---

## Experiment Flow

1. Load model
2. Measure memory usage
3. Run inference on same audio
4. Measure inference time
5. Calculate WER
6. Compare results across models

All models are tested on the **same audio file** to ensure fairness.

---

## Benchmark Results

| Model           | WER   | Inference Time | Memory Usage | Recommendation |
|----------------|-------|----------------|--------------|----------------|
| Distil-Whisper | High  | Moderate       | Low          | Moderate       |
| Whisper-Tiny   | Medium| Slow           | High         | Low            |
| Faster-Whisper | Low   | Fast           | Low          | ✅ Best        |

## 🏆 Key Findings

- **Whisper-Tiny**
  - Good accuracy
  - Slow CPU inference
  - High memory usage

- **Distil-Whisper**
  - Faster than Whisper-Tiny
  - Slightly worse accuracy
  - Good memory efficiency

- **Faster-Whisper**
  - Best accuracy (lowest WER)
  - Fastest inference
  - Lowest memory usage
  - Best suited for production

---


##  Usage (Demo)
```bash
# Run Distil-Whisper
python benchmark_distil_whisper.py

# Run Faster-Whisper
python benchmark_faster_whisper.py

# Run Whisper-Tiny
python STT.py

