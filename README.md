### 🔁 **Video-RAG Pipeline (Step-by-Step) - Approach 1**

#### 1. **Initialization**

   * `Config` ensures directories and device are set up.
   * `VideoRAGPipeline` instantiates all sub‑components:

     * `Transcriber` (ASR)
     * `SceneExtractor` (scene‑change frame sampling)
     * `FrameProcessor` (OCR, captions, detections, visual tags)
     * `Indexer` (embedding & HNSW index building)
     * `Retriever` (querying + LLM)

#### 2. **Run Pipeline** (`pipeline.run(query)`)

   1. **Frame Extraction**

      * `SceneExtractor.extract(video_path)`
      * Outputs: list of representative `frames` and their `times`.
   2. **Audio Transcription**

      * `Transcriber.transcribe(video_path)`
      * Outputs: full spoken text (`asr_text`) with caching.
   3. **Frame Processing**

      * `FrameProcessor.process(frames)`
      * For each frame:

        * **OCR** → extracts any visible text.
        * **Dense Captioning** (BLIP‑2 ViT‑G) → “Describe this image in detail.”
        * **Object Detection** (YOLOv8) → list of `(label, score)`.
        * **Visual Tags** (GroundingDINO + Tag2Text) → rich phrases per detected region.
      * Outputs: `ocr_texts`, `captions`, `detections`, `vis_tags`.
   4. **Index Building**

      * `Indexer.build(asr_text, ocr_texts, captions, vis_tags, frames)`
      * **Text embeddings** (Sentence‑Transformer) → HNSW index
      * **Image embeddings** (CLIP ViT-H) → HNSW index
      * Outputs: flat list `texts`, `text_index`, `img_index`.
   5. **Retriever Setup**

      * Instantiate `Retriever(texts, text_index, img_index, captions, detections, times)`.
   6. **Query & Generate**

      * `Retriever.query(query)`
      * Embed the user query → retrieve top text & image hits.
      * Build a combined prompt:

        ```
        [Text] …  
        [Vis t.s] dense caption  
        [Det t.s] object1(score), object2(score)  
        Q: <your query>  
        A:
        ```
      * Send to LLM (Groq API) → stream back `answer`.

#### 3. **Output**

   * Returns the full prompt + the LLM’s answer.

### Visual Flow (ASCII)

```
[ Video File ]  
      ↓  
[ SceneExtractor ] ──> frames, times  
      ↓  
[ Transcriber ] ──> asr_text  
      ↓  
[ FrameProcessor ] ───────────────────────────────────────────────┐
      ↓                                                           │
[ ocr_texts, captions, detections, vis_tags ]                     │
      ↓                                                           │
[ Indexer ] ──> text_index, img_index                             │
      ↓                                                           │
[ Retriever ] ──> embed(query) ──> retrieve top-K (text+images)   │
      ↓                                                           │
[ Build Prompt ] ──> “[Text]… [Vis]… [Det]… Q: <query> A:”        │
      ↓                                                           │
[ Groq LLM API ] ──> answer                                       │
      ↓                                                           │
[ Return prompt + answer ] <──────────────────────────────────────┘
```

- - - - 

### 🔁 **Video-RAG Pipeline (Step-by-Step) - Approach 2 (multimodal reasoning for both frame-specific queries and cross-frame specific( narative/long form) queries)**

#### 🛠 1. Offline Preprocessing (Build Your Video “Knowledge Base”)

1. **Scene‑Change Detection & Frame Extraction**

   * Run PySceneDetect on the raw video to find shot boundaries.
   * For each shot, grab one representative frame (e.g. the first frame of the shot).
   * Record its timestamp (in seconds or mm\:ss).

2. **Audio Transcription (ASR)**

   * Use Whisper to transcribe the entire video’s audio track into one text string.
   * Cache the result by hashing the video path.

3. **Frame‑Level Understanding**
   For each extracted frame:
   a. **OCR** (Tesseract or PaddleOCR) → any visible on‑screen text.
   b. **Dense Captioning** (BLIP‑2 ViT‑G) → a paragraph‑style caption: “A red butterfly rests on a green leaf…”
   c. **Object Detection** (YOLOv8) → list of `(label, confidence)` pairs.
   d. **Visual Tags**

   * Use GroundingDINO with the YOLO labels as prompts → bounding boxes for any named object.
   * Crop each box and feed to Tag2Text → richer phrases: “shimmering wing pattern,” “sunlit veins.”

4. **Embedding Extraction**

   * **Text Embeddings**: run Sentence‑Transformer on

     * the ASR text,
     * all OCR snippets,
     * all dense captions,
     * all visual‑tag phrases.
   * **Image Embeddings**: run CLIP (ViT‑H) on each frame.

5. **Index Construction**

   * Build two FAISS HNSW indexes (high‑recall k‑NN):

     1. **Text Index** ← text embeddings
     2. **Image Index** ← frame embeddings
   * Optionally pre‑cluster with k‑means to speed up construction.

6. **Caching**

   * Save all intermediate outputs (ASR text, captions, embeddings, indexes) under a hash key so you never recompute on the same video.

#### 🚀 2. Online Query & Answering

1. **User Submits Query**

   * Could be frame‑specific (“What color is the butterfly?”) or holistic (“Describe the sequence of events.”).

2. **Frame Retrieval**

   * Embed the query as both text (Sentence‑Transformer) and image (CLIP) vectors.
   * Search top K candidates in both indexes (e.g. K=5 for text, K=3 for images).
   * Merge/sort to select your final **set of K frames** most relevant to the query.

3. **Multimodal Reasoning**
   You have two modes, depending on your chosen model’s API:

   **A. Batch‑Image Prompting**

   ```python
   # If model supports N images + text at once:
   inputs = processor(images=[frame1,…,frameK],
                      text=user_query,
                      return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=256)
   answer  = processor.decode(outputs[0], skip_special_tokens=True)
   ```

   **B. Iterative Contextualization**

   1. Ask about frame₁ + query → get A₁
   2. Feed A₁ + frame₂ + same query → get A₂
   3. …continue through frame\_K
   4. Finally ask: “Combine A₁ … Aₖ into one coherent answer.”

4. **Answer Delivery**

   * The multimodal LLM returns a single text response, seamlessly integrating visual and temporal context.
   * No more stitching together OCR, captions, or tags by hand.

5. **Fallback (Optional)**

   * If the multimodal LLM fails or the query is purely textual, revert to the original Groq/text‑RAG step on your indexed captions & tags.

#### Visual Flow (ASCII)

```text
[ Video File ]
      ↓
1. Scene‑Change Detection
   • PySceneDetect finds key shot boundaries
   • Outputs: representative frames + timestamps

      ↓
2. Embedding & Retrieval
   • CLIP / Sentence‑Transformer embed frames & text (ASR, OCR, captions, tags)
   • Build HNSW indexes (FAISS) for fast k‑NN lookup

      ↓
3. Query Processing
   User submits “query” (frame‑specific or holistic)

      ↓
4. Top‑K Frame Retrieval
   • Embed query with CLIP / text encoder
   • Search both text‑index & image‑index to select top K frames

      ↓
5. Multimodal Reasoning
   ┌────────────────────────────────────────────────────────────┐
   │ Option A: Batch‑Image Prompting                            │
   │   • Processor packs K images + query into one input        │
   │   • Multimodal LLM (e.g. LLaVA, MiniGPT‑4) generates       │
   │     a single, cross‑frame answer                           │
   └────────────────────────────────────────────────────────────┘
   OR
   ┌────────────────────────────────────────────────────────────┐
   │ Option B: Iterative Contextualization                      │
   │   • Ask LLM about frame₁ + query → get A₁                  │
   │   • For each next frameᵢ: “Before you said A_{i–1}, now,   │
   │     with frameᵢ, what happens next?” → Aᵢ                  │
   │   • Finally: “Combine all into a coherent narrative.”      │
   └────────────────────────────────────────────────────────────┘

      ↓
6. Final Answer
   • Multimodal LLM returns a single, user‑friendly response
   • No manual prompt‑stitching of OCR/captions/tags

      ↓
7. (Optional) Text‑RAG Fallback
   • For purely textual queries or when multimodal fails
   • Use existing Groq/text‑RAG on concatenated captions + tags + OCR

      ↓
▶ User sees a natural, accurate answer to any video query  
```

This detailed flow ensures **any question**, whether about a **single frame** or **the entire sequence**, is handled smoothly by your unified, multimodal Video RAG system.

- - - - 

| Pipeline             | Core Innovation                           | Pros                               | Trade-offs                     |
| -------------------- | ----------------------------------------- | ---------------------------------- | ------------------------------ |
| **Video‑RAG**        | Fixed-time chunking + multimodal fine DAC | Simple to build, solid baseline    | Less flexible over formats     |
| **iRAG**             | Incremental/query-time ingestion          | Fast indexing, efficient use       | Query-time cost non-zero       |
| **SceneRAG**         | Narrative scene segmentation & graph      | Rich context continuity            | Scene detection needed         |
| **Omni‑AdaVideoRAG** | Adaptive retrieval scale per query        | Efficient & accurate for all cases | Needs intent classifier        |
| **VisRAG**           | Direct image-embedding without parsing    | Better visual grounding            | Lacks detailed text extraction |

- - - - 

| Pipeline             | Accuracy | Speed / Efficiency                         | Best For                                  |
| -------------------- | -------- | ------------------------------------------ | ----------------------------------------- |
| **SceneRAG**         | Highest  | Moderate to slow                           | Complex, long videos; multi-hop reasoning |
| **Omni-AdaVideoRAG** | High     | Efficient                                  | Versatile queries with varying complexity |
| **iRAG**             | Good     | Very fast ingestion, query-time extraction | Large-scale corpora, fast query needs     |

- - - - 

| **Pipeline**             | **Segmentation Approach**                        | **Retrieval Method**                       | **Core Innovation**                                                                    | **Code Availability**                |
| ------------------------ | ------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| **Video-RAG** (Leon1207) | Fixed-interval frames + ASR/OCR/object detection | Vector-based multimodal retrieval          | Lightweight, plug‑and‑play RAG for LVLMs using visually‑aligned text ([github.com][1]) | ✅ GitHub: Leon1207/Video‑RAG‑master  |
| **VideoRAG** (HKUDS)     | Fixed 30-sec clips + ImageBind + graph encoding  | Dual-channel: graph + visual embeddings    | Graph‑driven knowledge grounding + long‑video multimodal encoding                      | ✅ GitHub: HKUDS/VideoRAG             |
| **SceneRAG**             | Narrative-driven scene segmentation with LLM     | Scene-aware retrieval via knowledge graphs | Builds coherent scene graphs for multi-hop, long-range reasoning                       | ❌ No public code yet                 |
| **iRAG**                 | Minimal upfront indexing + deep query-time parse | Hybrid KNN over metadata + deep processing | On-demand extraction for fast indexing at scale                                        | ❌ No public code yet                 |
| **Omni‑AdaVideoRAG**     | Adaptive granularity (coarse/fine windows)       | Hierarchical text, visual, graph indices   | Query-guided scale adaptation with intent classification                               | ❌ No public code yet                 |

