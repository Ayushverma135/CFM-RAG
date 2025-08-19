# CFM-RAG

**Cross-Frame Multimodal Retrieval-Augmented Generation for Video Understanding**

> CFM-RAG brings cross-frame audio, text and visual evidence together to enable accurate, context-rich video Q\&A and content search.

---

## Overview

This repository implements a **video Retrieval-Augmented Generation (RAG)** pipeline that:

* Extracts representative frames from a video.
* Transcribes audio to text (ASR).
* Performs OCR, object detection, and segmentation on frames.
* Generates frame captions.
* Builds CLIP embeddings for both text and images.
* Retrieves the most relevant text and frames for a user query.
* Constructs a context-rich prompt and queries an LLM to produce a final answer.

The entrypoint is `src/main.py`, which instantiates `VideoRAGPipeline` and runs a sample query (e.g., “How many tin cans are there in the video?”).

---

## Features

* Cross-frame fusion of multimodal signals (ASR, OCR, captions, detections) for robust retrieval.
* Timestamped evidence (captions + detection labels) attached to retrieval results for traceability.
* Per-item caching to speed up repeated runs and reduce redundant model calls.
* Streaming LLM integration for low-latency, traceable answers.

---

## High-level flow

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4c7dc859-727c-43b5-8774-329ed1213991" />


1. **Scene extraction** (`SceneExtractor.extract`)

   * Uses PySceneDetect to detect cuts and Decord to load key frames.
   * Caches scenes keyed by video content hash.
   * Returns arrays of `frames` and `times`.

2. **Transcription** (`Transcriber.transcribe`)

   * Uses FFmpeg to extract audio (16 kHz mono float32) and Whisper Base for ASR.
   * Caches transcripts using SHA256 of the file path.

3. **Frame processing** (`FrameProcessor.process`)

   * Captions: GPU-batched BLIP-2 (FLAN-T5-XL) for all frames.
   * Per-frame (parallel): Tesseract OCR, YOLOv8x detection, GroundingDINO tiny + SAM huge for segmentation.
   * Caches results per frame content hash.
   * Returns `(ocr_texts, captions, detections, masks)`.

4. **Indexing** (`Indexer.build`)

   * Builds open\_clip ViT-H-14 embeddings for text (ASR + OCR + captions) and for images (frames).
   * Uses per-item caching (text hashed by content, images by frame hash).
   * Returns `(texts, text_embs, img_embs)`.

5. **Retrieval + LLM** (`Retriever.query`)

   * Encodes the query with CLIP and computes cosine similarity with text and image embeddings.
   * Selects top-3 text and top-3 image hits (configurable).
   * Builds a prompt including: top text snippets, timestamped image captions, and detection labels with confidences.
   * Calls Groq Chat Completions (model: `meta-llama/llama-4-scout-17b-16e-instruct`) using streaming.
   * Returns `(prompt, answer)`.

---

## Components

* `src/main.py` — Minimal entrypoint; creates pipeline with `Config.VIDEO_PATH` and prints prompt/answer.
* `src/pipeline.py` — `VideoRAGPipeline` orchestrator; validates config and ensures `Config.CACHE_DIR` exists.
* `src/scene_extractor.py` — PySceneDetect + Decord scene/key-frame extraction (cache by file-hash).
* `src/transcriber.py` — FFmpeg audio extraction + Whisper transcription (cache by path-hash).
* `src/frame_processor.py` — BLIP-2 captioning + per-frame OCR/detection/segmentation in a process pool (cache by frame-hash).
* `src/indexer.py` — open\_clip ViT-H-14 embeddings for texts and images with per-item caching.
* `src/retriever.py` — CLIP-based retrieval across text & image embeddings and Groq LLM call.
* `src/cache_utils.py` — Simple pickle-based cache utilities (store in `Config.CACHE_DIR`).
* `src/hash_utils.py` — SHA256 helpers for files and frames.
* `src/config.py` — Central configuration (paths, device selection, thresholds).

---

## Caching strategy

* **Scene list & key frames:** keyed by video file content hash.
* **Transcription:** keyed by SHA256 of the file path (note: may go stale if path remains the same but content changes).
* **Per-frame results (OCR/detections/masks):** keyed by frame content hash.
* **Text embeddings:** keyed by text SHA256.
* **Image embeddings:** keyed by frame content hash.
* Cache files are stored as `.pkl` files inside `Config.CACHE_DIR`.

---

## Models & Dependencies

* **Video processing:** PySceneDetect, Decord
* **ASR:** Whisper Base (HF Transformers)
* **Vision:**

  * Captions: BLIP-2 (FLAN-T5-XL) — GPU recommended
  * OCR: Tesseract (`pytesseract`) — system Tesseract required
  * Detection: YOLOv8x (`ultralytics`)
  * Grounded detection: GroundingDINO tiny
  * Segmentation: SAM huge
* **Embeddings:** open\_clip ViT-H-14 (`laion2b_s32b_b79k`)
* **LLM:** Groq Chat Completions (`meta-llama/llama-4-scout-17b-16e-instruct`)
* **Utilities:** Torch, PIL/Pillow, NumPy, FFmpeg

> **Note:** Large models (BLIP-2 XL, SAM huge, CLIP ViT-H-14) require significant GPU memory. Consider lighter alternatives for constrained hardware.

---

## Configuration

Edit `src/config.py` or set environment variables as needed:

* `Config.VIDEO_PATH` — Absolute path to input video.
* `Config.CACHE_DIR` — Absolute path for cache storage.
* `Config.DEVICE` — `"cuda"` if CUDA is available, otherwise `"cpu"`.
* `GROQ_API_KEY` — Must be set in the environment for the Retriever to call Groq. Replace any hardcoded default in code with a secure env variable.

---

## Runtime behavior & output

The pipeline prints progress across stages:

```
Extract scenes → Transcribe → Process frames → Build index → Initialize retriever → Query retriever
```

On completion it prints:

* The constructed retrieval-augmented prompt used for the LLM.
* The LLM-generated answer (streaming supported).

---

## Operational notes

* Heavy GPU/VRAM demand for certain models. CPU fallback is available but slow.
* Windows-specific notes:

  * Ensure `ffmpeg` and `tesseract` are installed and added to `PATH`.
  * Large HF model downloads happen on first run.
  * Using a `ProcessPoolExecutor` with GPU models requires care: workers load models on demand and may increase memory footprint.
* Replace any hardcoded Groq API key with `GROQ_API_KEY`.

---

## Limitations & Risks

* Scene extraction uses only scene-boundary key frames — events inside long scenes can be missed.
* Retrieval uses a simple top-k fusion across text & image scores (no learned reranker or joint reranking).
* Caching design has weaknesses:

  * Transcription cache keyed by path can go stale if video content changes but path doesn’t.
  * Pickle-based cache has no file-locking; concurrent runs may race or corrupt cache.
* Model and runtime size: memory pressure and long cold-starts are expected on first run.
* Hardcoded absolute paths reduce portability — prefer environment-configured paths.

---

## How to run

1. Install Python packages (example):

```bash
pip install decord torch numpy Pillow transformers>=4.30.1 segment_anything ultralytics pytesseract soundfile ffmpeg-python groq sentence-transformers open-clip-torch faiss-cpu scenedetect[opencv] opencv-python shapely pycocotools
```

2. System binaries (if you haven't yet):

```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender1 \
    git
```

3. Configure `src/config.py` or export env vars:

```bash
export GROQ_API_KEY="<your_groq_api_key>"
# or edit Config.VIDEO_PATH and Config.CACHE_DIR in src/config.py
```

4. Run the demo:

```bash
python src/main.py
```

5. To change the query, edit `src/main.py` and re-run.

---

## Tips & improvements

* Use lighter models for constrained environments: BLIP-2 smaller, SAM-base, YOLOv8n, etc.
* Improve caching by hashing file contents rather than file paths for ASR.
* Add file-locking or a more robust key-value store (Redis/LMDB) to avoid pickle races.
* Add a reranker that jointly considers text + visual features or finetune a lightweight cross-encoder for improved precision.
* Extract and store short video clips for top visual hits to provide direct visual evidence for answers.

---

## Contributing

Contributions welcome — open issues or pull requests for bug fixes, performance improvements, and new features.
