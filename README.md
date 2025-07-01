### 🔁 **Video-RAG Pipeline (Step-by-Step)**

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

---

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

