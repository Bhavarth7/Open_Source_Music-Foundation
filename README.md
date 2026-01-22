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


 Additional Info
 [7 tools called]

Here’s a detailed “engineer’s guide” to this repo: what it does, how it’s built, and how the code fits together.

## What HeartMuLa Is
- Open-source music foundation system for text/lyrics-to-music generation with multilingual support.
- Components: a music language model (HeartMuLa), an audio codec/vocoder (HeartCodec), a Whisper-based lyrics transcriber (HeartTranscriptor), and CLAP-style audio–text alignment (HeartCLAP).

## High-Level Architecture
1. **Text/Tag Encoding**: Lyrics + style tags tokenized with a shared tokenizer (`tokenizer.json`). BOS/EOS enforced; tags wrapped in `<tag>…</tag>`.
2. **Music LM (HeartMuLa)**: Transformer decoder (torchtune LLaMA 3.2 variants) generates discrete audio codes (multi-codebook RVQ tokens).
3. **Guidance**: Classifier-free guidance (CFG) duplicates batch when `cfg_scale > 1.0`.
4. **Autoregressive Loop**: Generates codebook frames step-by-step with KV-cache reuse.
5. **Codec/Vocoder (HeartCodec)**: Flow-matching + scalar codec decodes discrete codes to waveform; overlap-add smoothing for long audio.
6. **Transcription**: Optional Whisper pipeline for lyrics transcription.

## Folder Structure (important bits)
- `examples/`
  - `run_music_generation.py`: CLI entrypoint for generation.
  - `run_lyrics_transcription.py`: CLI for transcription.
- `src/heartlib/`
  - `__init__.py`: exports pipelines.
  - `pipelines/music_generation.py`: end-to-end generation pipeline (preprocess → forward → postprocess).
  - `pipelines/lyrics_transcription.py`: Whisper ASR pipeline loader.
  - `heartmula/`
    - `configuration_heartmula.py`: config (backbone/decoder flavors, vocab sizes, codebooks, MUQ dim).
    - `modeling_heartmula.py`: music LM definitions (LLaMA flavors, sampling, CFG, KV-cache setup).
  - `heartcodec/`
    - `configuration_heartcodec.py`: RVQ + flow-matching + scalar codec params.
    - `modeling_heartcodec.py`: codec forward/detokenize pipeline with overlap-add.
    - `models/flow_matching.py`, `sq_codec.py`, `transformer.py`: codec internals.

## Key Components & Techniques
### HeartMuLa (music LM) — `heartmula/modeling_heartmula.py`
- Uses torchtune LLaMA 3.2 decoder variants (3B, 7B, 300M, 400M) via a flavor map.
- Dual stacks: `backbone` and `decoder`; text embeddings projected into decoder space.
- Audio embeddings: RVQ-style, multiple codebooks; `audio_head` produces logits for codebooks.
- MUQ embedding (`muq_linear`) for continuous conditioning (placeholder here, zeroed if none).
- CFG: batch is doubled when `cfg_scale > 1`; unconditional branch uses `unconditional_text_embedding`.
- Sampling: top-k + temperature with a multinomial sampler that avoids CUDA sync.
- KV cache: `setup_caches(max_batch_size)` preallocates caches and causal masks for backbone/decoder.
- Generation: `generate_frame` consumes prompt tokens+mask+pos and outputs next frame (per codebook).
- Autocast with configurable `dtype` (bfloat16 in examples).

### Pipeline — `pipelines/music_generation.py`
- Preprocess:
  - Reads tags/lyrics from file or string; enforces BOS/EOS and tag wrappers.
  - Builds a `tokens` matrix shaped `(prompt_len, parallel_num)` where last column is text; other columns reserved for audio codebooks.
  - CFG-aware duplication of tensors when `cfg_scale != 1`.
- Forward:
  - Prepares model caches, runs initial frame generation, then autoregressive loop for up to `max_audio_length_ms // 80` frames.
  - Early stops on `audio_eos_id`.
  - Pads audio tokens per step to align with multi-codebook layout.
- Postprocess:
  - Calls `HeartCodec.detokenize` to turn codebooks into waveform.
  - Saves with `torchaudio.save` at 48 kHz.
- Loader:
  - `from_pretrained` expects `HeartCodec-oss/`, `HeartMuLa-oss-{version}/`, `tokenizer.json`, `gen_config.json`.
  - Optional quantization via `BitsAndBytesConfig` if provided.

### HeartCodec — `heartcodec/modeling_heartcodec.py`
- Combines:
  - FlowMatching RVQ for discrete codes.
  - ScalarModel decoder (in `sq_codec.py`) for waveform synthesis.
- `detokenize`:
  - Repeats codes to satisfy minimum length; overlap-add with a Hann-like window to avoid seams.
  - Generates latents per hop, decodes to audio, trims to target length.
  - Uses guidance scale and configurable inference steps.

### Lyrics Transcription — `pipelines/lyrics_transcription.py`
- Thin wrapper over `AutomaticSpeechRecognitionPipeline` with Whisper; loads from `HeartTranscriptor-oss` checkpoint; chunked ASR with batch size 16.

## Data Flow (generation)
`lyrics/tags → tokenizer → prompt tokens → HeartMuLa (CFG, KV cache) → discrete codebooks → HeartCodec (flow-matching + scalar codec) → waveform`

## Important Parameters (examples/run_music_generation.py)
- `--max_audio_length_ms`: main lever for memory/time.
- `--topk`, `--temperature`, `--cfg_scale`: sampling controls (CFG doubles batch).
- `dtype`: bfloat16 default; configurable in code.
- `device`: set to CUDA in script; can be adapted to CPU/offload if needed.

## Notable Engineering Choices
- **Torchtune LLaMA 3.2 backbones** for open weights + strong decoding speed.
- **CFG batch-doubling** for controllability with minimal code.
- **Multi-codebook RVQ** for efficient discrete audio representation.
- **Overlap-add vocoder stitching** to reduce boundary artifacts.
- **Autocast + KV caches** to keep generation feasible on GPU.
- **BitsAndBytes hook** in loader for potential 8-bit loading (when supported by checkpoints).

## How to Run (quick)
- Music gen: `python ./examples/run_music_generation.py --model_path=./ckpt --version="3B" --lyrics=./assets/lyrics.txt --tags=./assets/tags.txt --save_path=./assets/output.mp3`
- Transcription: `python ./examples/run_lyrics_transcription.py --audio_path=./audio.mp3 --output_path=./transcription.txt`

## Practical Notes for MLEs
- VRAM: 3B model typically needs ≥16 GB; reduce `--max_audio_length_ms` or use CPU/offload if tight.
- CFG doubles memory; set `cfg_scale=1.0` to save VRAM.
- Quantization: `BitsAndBytesConfig` can be passed to `HeartMuLa.from_pretrained`; ensure checkpoints/transformers version support it.
- Latency: `max_audio_length_ms` scales linearly with generation time; overlap-add helps quality at shorter hops.
- Tokenizer: shared for tags/lyrics; ensure BOS/EOS are present (pipeline enforces).
- Checkpoint layout: `./ckpt/HeartMuLa-oss-{version}`, `./ckpt/HeartCodec-oss`, plus `tokenizer.json`, `gen_config.json`.

If you want, I can draft a MARKDOWN “Deep Dive” doc inside `docs/` or extend the README with an architecture section tailored for MLEs.