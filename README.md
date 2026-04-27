# 🎙️ Speech-to-Reasoning Pipeline
### Whisper ASR → Quantized LLM (Unsloth Dynamic 4-bit)

A complete end-to-end pipeline that transcribes **spoken audio into text** using OpenAI's Whisper, then feeds that transcription into a **quantized reasoning LLM** (Qwen2.5-7B via Unsloth) to generate step-by-step logical answers — all running in a single Google Colab notebook on a free T4 GPU.

---

## 🔁 Pipeline Architecture

```
Audio File (WAV / MP3 / M4A / FLAC)
         │
         ▼
  ┌──────────────┐
  │  OpenAI      │  whisper.load_model() + model.transcribe()
  │  Whisper ASR │  → Transcribed text + timestamps
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Prompt     │  apply_chat_template()
  │   Template   │  → System message + user query
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Unsloth     │  FastLanguageModel.from_pretrained()
  │  4-bit LLM   │  Qwen2.5-7B-Instruct (dynamic 4-bit)
  └──────┬───────┘
         │
         ▼
    Reasoned Answer
```

---

## ✨ Features

- 🎤 **Automatic Speech Recognition** — Whisper `small` / `medium` / `large-v3` support
- 🤖 **4-bit Quantized LLM** — ~60% less VRAM vs float16, 2× faster inference via Unsloth
- 🧠 **Step-by-step reasoning** — structured system prompt elicits logical, detailed answers
- 📦 **Batch processing** — run multiple audio files in a single loop
- 📁 **Upload your own audio** — supports WAV, MP3, M4A, FLAC, OGG
- 📊 **Performance metrics** — tokens/sec, VRAM usage, inference time reported automatically
- 🔄 **Reusable helper function** — `speech_to_reasoning(audio_path)` wraps the full pipeline
- 🌐 **Multi-language ready** — remove `language="en"` for auto language detection

---

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Important:** Set runtime to GPU before running.  
> `Runtime → Change runtime type → T4 GPU`

### 2. Run all cells in order

The notebook is fully self-contained. Cell 3 auto-generates a sample audio file using gTTS so you can run the entire pipeline without uploading anything.

### 3. Upload your own audio (optional)

Run **Step 12** cell and click **"Choose Files"** to upload your own `.wav` or `.mp3` file.

---

## 📋 Notebook Structure

| Step | Description |
|------|-------------|
| 1 | Install all dependencies |
| 2 | Imports & GPU verification |
| 3 | Generate sample audio query (gTTS) |
| 4 | Load Whisper & transcribe audio |
| 5 | Inspect segment-level output |
| 6 | Load quantized LLM via Unsloth |
| 7 | Build reasoning prompt |
| 8 | Run inference & generate answer |
| 9 | Display full pipeline results |
| 10 | Reusable `speech_to_reasoning()` function |
| 11 | Batch processing demo |
| 12 | Upload your own audio file |
| 13 | GPU memory cleanup |

---

## 🤖 Supported Models

### Whisper (ASR)

| Size | VRAM | Speed | Accuracy |
|------|------|-------|----------|
| `tiny` | ~1 GB | Fastest | Basic |
| `base` | ~1 GB | Fast | Good |
| `small` | ~2 GB | Moderate | **Default** |
| `medium` | ~5 GB | Slow | Better |
| `large-v3` | ~10 GB | Slowest | Best |

### LLM (Reasoning)

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | ~3 GB | Lightweight, fast |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | ~6 GB | **Default** — balanced |
| `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | ~7 GB | Strong reasoning |
| `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | ~12 GB | Best accuracy |

---

## ⚙️ Configuration

Key variables at the top of each relevant cell:

```python
WHISPER_MODEL_SIZE = "small"      # Change to "medium" for better accuracy
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
LOAD_IN_4BIT = True
```

Generation parameters:

```python
GEN_CONFIG = dict(
    max_new_tokens  = 512,
    temperature     = 0.3,   # Lower = more deterministic
    top_p           = 0.9,
    repetition_penalty = 1.1,
    do_sample       = True,
)
```

---

## 💡 Usage Example

```python
# Run on any audio file
result = speech_to_reasoning(
    audio_path     = "my_question.mp3",
    whisper_size   = "small",
    max_new_tokens = 512,
    temperature    = 0.3,
)

print(result["transcription"])  # What Whisper heard
print(result["response"])       # LLM's reasoned answer
print(result["metrics"])        # tokens/sec, VRAM, timing
```

---

## 🧠 Memory Management

The pipeline frees Whisper from GPU VRAM before loading the LLM to avoid OOM errors on a 16 GB T4:

```python
del whisper_model
gc.collect()
torch.cuda.empty_cache()
```

This is done automatically by the `speech_to_reasoning()` helper function.

---

## 📁 Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MP3 | `.mp3` | Most common, works great |
| WAV | `.wav` | Uncompressed, best quality |
| M4A | `.m4a` | iPhone voice memos |
| FLAC | `.flac` | Lossless audio |
| OGG | `.ogg` | Open format |

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:

- `openai-whisper` — ASR model
- `unsloth` — quantized LLM loading & fast inference
- `transformers` — tokenizer & HuggingFace model hub
- `torch` — GPU tensor operations
- `bitsandbytes` — 4-bit quantization backend
- `accelerate` — multi-GPU / device mapping
- `gtts` — demo audio synthesis
- `soundfile`, `librosa` — audio file handling
- `ffmpeg` (system) — audio decoding

---

## 🛠️ Requirements

- Google Colab with **T4 GPU** (free tier) or better
- Python 3.10+
- CUDA 11.8+ (handled automatically by Colab)

---

## 📄 License

This project is for educational purposes. Model weights are subject to their respective licenses:
- Whisper: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)
- Qwen2.5: [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Llama 3: [Meta Llama 3 License](https://llama.meta.com/llama3/license/)
