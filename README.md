# HeartMuLa: Open-Source Music Foundation Models

**HeartMuLa** is an open-source family of music foundation models for controllable music generation with multilingual lyrics support. This repository provides production-ready implementations of music language models, audio codecs, and transcription systems.

## Overview

HeartMuLa enables text-conditioned music generation through a modular architecture combining:

- **HeartMuLa**: Transformer-based music language model generating discrete audio tokens conditioned on lyrics and style tags
- **HeartCodec**: 12.5 Hz neural audio codec with high-fidelity reconstruction
- **HeartTranscriptor**: Whisper-based lyrics transcription model
- **HeartCLAP**: Cross-modal audio-text alignment for music retrieval

## Key Features

- **Multilingual Support**: Generates music conditioned on lyrics in English, Chinese, Japanese, Korean, Spanish, and more
- **Controllable Generation**: Fine-grained control via style tags (genre, mood, instruments)
- **High Fidelity**: 12.5 Hz codec maintains audio quality at low bitrates
- **Production Ready**: Optimized inference pipeline with GPU/CPU support

## Technical Architecture

The system employs a two-stage generation pipeline:
1. **Language Model Stage**: HeartMuLa generates discrete audio tokens from text inputs using a transformer decoder architecture
2. **Vocoder Stage**: HeartCodec decodes tokens to waveform audio using flow matching and vector quantization

The 3B parameter model achieves real-time generation (RTF ≈ 1.0) on modern GPUs.

## Installation

```bash
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
pip install -e .
```

## Model Weights

Download pretrained checkpoints:

```bash
# HuggingFace
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss'



Expected checkpoint structure:
```
./ckpt/
├── HeartCodec-oss/
├── HeartMuLa-oss-3B/
├── gen_config.json
└── tokenizer.json
```

## Usage

### Music Generation

```bash
python ./examples/run_music_generation.py \
    --model_path=./ckpt \
    --version="3B" \
    --lyrics=./assets/lyrics.txt \
    --tags=./assets/tags.txt \
    --save_path=./assets/output.mp3 \
    --max_audio_length_ms=240000 \
    --topk=50 \
    --temperature=1.0 \
    --cfg_scale=1.5
```

**Parameters:**
- `--model_path`: Path to pretrained checkpoints (required)
- `--version`: Model version (`3B` or `7B`)
- `--lyrics`: Path to lyrics file with section markers (`[Verse]`, `[Chorus]`, etc.)
- `--tags`: Comma-separated style tags (e.g., `piano,happy,romantic`)
- `--max_audio_length_ms`: Maximum generation length in milliseconds
- `--topk`: Top-k sampling parameter
- `--temperature`: Sampling temperature
- `--cfg_scale`: Classifier-free guidance scale
