# yt-whisper 🎙️

A toolkit for building custom speech datasets from YouTube videos and fine-tuning OpenAI's [Whisper](https://github.com/openai/whisper) model on an Apple Silicon Mac (M1/M2/M3).

## Overview

| Script | Purpose |
|---|---|
| [`youtube_dataset_builder.py`](#1-youtube_dataset_builderpy) | Download YouTube audio + subtitles → aligned training dataset |
| [`train_whisper_lora.py`](#2-train_whisper_lorapy) | Fine-tune Whisper with LoRA on M1 GPU |
| [`live_translate_m1.py`](#3-live_translate_m1py) | Real-time mic/system-audio transcription & translation (M1-optimized) |
| [`live_translate.py`](#4-live_translatepy) | Real-time mic transcription to English (CPU/any platform) |

---

## Setup

```bash
# Clone and enter the project
cd yt-whisper

# Create and activate virtual environment (Apple Silicon)
python3 -m venv venv_arm64
source venv_arm64/bin/activate

# Install dependencies
pip install yt-dlp torch transformers datasets peft sounddevice numpy
pip install mlx-whisper          # for live_translate_m1.py
pip install openai-whisper        # for live_translate.py
pip install argostranslate        # for non-English output in live_translate_m1.py
```

> [!NOTE]
> `ffmpeg` must also be installed: `brew install ffmpeg`

---

## 1. `youtube_dataset_builder.py`

Downloads audio and subtitles from a YouTube video and slices them into aligned `(audio_clip, sentence)` pairs ready for Whisper fine-tuning.

**Outputs:**
- `dataset/audio/clip_NNNN.wav` — 4–18 second mono 16 kHz audio clips
- `dataset/train.jsonl` — primary dataset (id, audio path, timestamps, sentence)
- `dataset/train.csv` — same data in CSV format
- `dataset/train_target.jsonl` — *(optional)* paired target-language sentences for translation verification

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `url` | *(positional)* | — | YouTube video URL |
| `--lang` | `-l` | `en` | Source subtitle language code (`en`, `de`, `es`, …) |
| `--output-dir` | `-o` | `dataset` | Output directory |
| `--target-lang` | — | `None` | Secondary subtitle language for verification (`es`, `zh`, …) |
| `--allow-auto` | — | off | Accept YouTube auto-generated captions (lower quality) |
| `--cookies` | — | `None` | Path to Netscape cookies file (for age-restricted videos) |
| `--proxy` | — | `None` | Proxy URL passed to yt-dlp |
| `--user-agent` | — | `None` | Custom user-agent string passed to yt-dlp |

### Examples

**Basic — English video, build transcription dataset:**
```bash
python youtube_dataset_builder.py "https://www.youtube.com/watch?v=VIDEO_ID" -l en
```

**German video with auto-captions allowed:**
```bash
python youtube_dataset_builder.py "https://www.youtube.com/watch?v=VIDEO_ID" -l de --allow-auto
```

**English source + Spanish verification (for translation training):**
```bash
python youtube_dataset_builder.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  -l en --target-lang es
```
This produces `train.jsonl` (English sentences) **and** `train_target.jsonl` (aligned Spanish sentences).

**With cookies (for restricted content):**
```bash
# Export cookies from your browser as cookies.txt, then:
python youtube_dataset_builder.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  -l en --cookies cookies.txt
```

> [!TIP]
> The script only uses **creator-uploaded subtitles** by default. These are much more accurate than auto-generated captions. Pass `--allow-auto` to fall back to auto-generated captions when manual ones aren't available.

> [!NOTE]
> If YouTube blocks yt-dlp with HTTP 400 errors, try updating it first: `pip install -U yt-dlp`

---

## 2. `train_whisper_lora.py`

Fine-tunes a Whisper model using [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) — a parameter-efficient method that only trains a small set of adapter weights rather than the full model. Uses Apple MPS (Metal) acceleration automatically.

**Outputs:**
- `whisper-lora-adapter/` — saved LoRA adapter weights and processor config

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--dataset-dir` | `-d` | `dataset` | Path to dataset directory (must contain `train.jsonl`) |
| `--model` | `-m` | `openai/whisper-small` | Hugging Face base model ID |
| `--output-dir` | `-o` | `whisper-lora-adapter` | Where to save the trained LoRA adapter |
| `--epochs` | `-e` | `3` | Number of training epochs |
| `--lang` | `-l` | `german` | Target language name (full word, e.g. `english`, `spanish`) |
| `--task` | `-t` | `translate` | `translate` (speech → English) or `transcribe` (speech → same language) |

### Available base models

| Model ID | Size | Speed | Accuracy |
|---|---|---|---|
| `openai/whisper-tiny` | ~39M params | Fastest | Lowest |
| `openai/whisper-base` | ~74M params | Fast | Low |
| `openai/whisper-small` | ~244M params | Moderate | Good |
| `openai/whisper-medium` | ~769M params | Slow | Better |
| `openai/whisper-large-v3` | ~1.5B params | Slowest | Best |

### Examples

**Fine-tune on German speech → English translation, 3 epochs:**
```bash
python train_whisper_lora.py -d dataset -m openai/whisper-small -l german -t translate -e 3
```

**Fine-tune on Spanish transcription (speech → Spanish text), 5 epochs:**
```bash
python train_whisper_lora.py -d dataset -m openai/whisper-small -l spanish -t transcribe -e 5
```

**Use a larger model with a custom output path:**
```bash
python train_whisper_lora.py \
  -d dataset \
  -m openai/whisper-medium \
  -l english \
  -t transcribe \
  -e 5 \
  -o my-english-adapter
```

> [!NOTE]
> `--lang` uses full language names (e.g. `german`, `spanish`, `english`) not codes — this is how Whisper's tokenizer expects them.

> [!TIP]
> Start with `whisper-small` and 3 epochs. If the dataset is large (500+ clips), try `whisper-medium` for better quality. LoRA trains only ~1% of parameters, so memory usage is low even on 8 GB M1.

---

## 3. `live_translate_m1.py`

Real-time speech transcription and translation using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon GPU-optimized). Captures audio from your microphone or system audio via **BlackHole** virtual device.

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input-language` | `-i` / `-l` | `auto` | Spoken language code (`en`, `de`, `es`, `fr`, …). `auto` = Whisper detects it |
| `--output-language` | `-o` | `en` | Target language for output. Use `orig` to output the original transcript unchanged |
| `--dual` | — | off | Show both the original spoken text AND the translation side-by-side |
| `--model` | `-m` | `turbo` | Model size: `turbo` (best), `small`, `base`, `tiny` |

### Models

| Value | Hugging Face repo | Notes |
|---|---|---|
| `turbo` | `mlx-community/whisper-large-v3-turbo` | Best accuracy, recommended |
| `small` | `mlx-community/whisper-small-mlx` | Faster, good for weaker hardware |
| `base` | `mlx-community/whisper-base-mlx` | Very fast, lower accuracy |
| `tiny` | `mlx-community/whisper-tiny` | Fastest, minimal accuracy |

### Examples

**Auto-detect language, translate to English:**
```bash
python live_translate_m1.py
```

**Translate German speech to English:**
```bash
python live_translate_m1.py -i de -o en
```

**Show dual view (original + translation) for Japanese speech:**
```bash
python live_translate_m1.py -i ja -o en --dual
```

**Just transcribe in the original language (no translation):**
```bash
python live_translate_m1.py -i de -o orig
```

**Translate to Spanish (uses Argos Translate for non-English output):**
```bash
python live_translate_m1.py -i en -o es
```

**Use a smaller/faster model:**
```bash
python live_translate_m1.py -i de -o en -m small
```

> [!NOTE]
> To capture system audio (e.g. from a browser tab) instead of the mic, install [BlackHole](https://existential.audio/blackhole/) and set it as your system audio output. The script auto-detects it.

> [!TIP]
> Use `--dual` when learning a language — you'll see the spoken sentence in its original language alongside the English translation.

---

## 4. `live_translate.py`

A simpler, CPU-compatible live transcription script using the standard `openai-whisper` library. Always outputs English. Good fallback if you're not on Apple Silicon.

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--language` | `-l` | auto-detect | Input spoken language code (`de`, `es`, `fr`, `ja`, `zh`, …) |
| `--model` | `-m` | `base` | Model size: `tiny`, `base`, or `small` |

### Examples

**Auto-detect and translate to English:**
```bash
python live_translate.py
```

**Specify German input, use small model:**
```bash
python live_translate.py -l de -m small
```

---

## Typical Workflow

```
1. Find a YouTube video with creator-uploaded subtitles in your target language
        ↓
2. Build the dataset
   python youtube_dataset_builder.py "<URL>" -l de -o dataset
        ↓
3. Fine-tune Whisper with LoRA
   python train_whisper_lora.py -d dataset -m openai/whisper-small -l german -t transcribe -e 3
        ↓
4. Use live_translate_m1.py with your fine-tuned adapter for real-time inference
```

---

## Project Structure

```
yt-whisper/
├── youtube_dataset_builder.py   # YouTube → training dataset
├── train_whisper_lora.py        # LoRA fine-tuning
├── live_translate_m1.py         # Real-time translation (M1 GPU)
├── live_translate.py            # Real-time translation (CPU)
├── test_mic_permissions.py      # macOS mic permission helper
├── dataset/                     # Generated by youtube_dataset_builder
│   ├── train.jsonl
│   ├── train.csv
│   ├── train_target.jsonl       # Optional: target-language verification
│   └── audio/
│       ├── clip_0001.wav
│       └── ...
└── whisper-lora-adapter/        # Generated by train_whisper_lora
    └── ...
```
