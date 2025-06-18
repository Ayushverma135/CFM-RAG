### 🔁 **Video-RAG Pipeline (Step-by-Step)**

#### 1. **Preprocessing**

* **Frame Extraction**: Sample frames every few seconds from the video.
* **Audio Transcription (ASR)**: Use models like Whisper to transcribe speech to text.
* **OCR (Optional)**: Extract on-screen text using Tesseract.
* **Image Captioning**: Generate visual descriptions using BLIP-2 or similar.

#### 2. **Multimodal Chunking**

* Convert extracted elements (ASR, OCR, captions) into textual "chunks" with timestamps.

#### 3. **Embedding & Indexing**

* Convert chunks into vector embeddings using:

  * **Sentence Transformers** (for text)
  * **CLIP/OpenCLIP** (for frames)
* Store embeddings in a **vector store** (e.g., FAISS).

#### 4. **Query & Retrieval**

* User inputs a question → embed it → retrieve top-K relevant chunks from the vector DB.

#### 5. **Prompt Construction**

* Combine retrieved chunks into a structured prompt with temporal/visual context.

#### 6. **Answer Generation**

* Send the prompt to an LLM (e.g., **Groq**, **GPT-4**, **Claude**) to generate a final answer.




| Pipeline             | Core Innovation                           | Pros                               | Trade-offs                     |
| -------------------- | ----------------------------------------- | ---------------------------------- | ------------------------------ |
| **Video‑RAG**        | Fixed-time chunking + multimodal fine DAC | Simple to build, solid baseline    | Less flexible over formats     |
| **iRAG**             | Incremental/query-time ingestion          | Fast indexing, efficient use       | Query-time cost non-zero       |
| **SceneRAG**         | Narrative scene segmentation & graph      | Rich context continuity            | Scene detection needed         |
| **Omni‑AdaVideoRAG** | Adaptive retrieval scale per query        | Efficient & accurate for all cases | Needs intent classifier        |
| **VisRAG**           | Direct image-embedding without parsing    | Better visual grounding            | Lacks detailed text extraction |


| Pipeline             | Accuracy | Speed / Efficiency                         | Best For                                  |
| -------------------- | -------- | ------------------------------------------ | ----------------------------------------- |
| **SceneRAG**         | Highest  | Moderate to slow                           | Complex, long videos; multi-hop reasoning |
| **Omni-AdaVideoRAG** | High     | Efficient                                  | Versatile queries with varying complexity |
| **iRAG**             | Good     | Very fast ingestion, query-time extraction | Large-scale corpora, fast query needs     |

| **Pipeline**             | **Segmentation Approach**                        | **Retrieval Method**                       | **Core Innovation**                                                                    | **Code Availability**                |
| ------------------------ | ------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| **Video-RAG** (Leon1207) | Fixed-interval frames + ASR/OCR/object detection | Vector-based multimodal retrieval          | Lightweight, plug‑and‑play RAG for LVLMs using visually‑aligned text ([github.com][1]) | ✅ GitHub: Leon1207/Video‑RAG‑master  |
| **VideoRAG** (HKUDS)     | Fixed 30-sec clips + ImageBind + graph encoding  | Dual-channel: graph + visual embeddings    | Graph‑driven knowledge grounding + long‑video multimodal encoding                      | ✅ GitHub: HKUDS/VideoRAG             |
| **SceneRAG**             | Narrative-driven scene segmentation with LLM     | Scene-aware retrieval via knowledge graphs | Builds coherent scene graphs for multi-hop, long-range reasoning                       | ❌ No public code yet                 |
| **iRAG**                 | Minimal upfront indexing + deep query-time parse | Hybrid KNN over metadata + deep processing | On-demand extraction for fast indexing at scale                                        | ❌ No public code yet                 |
| **Omni‑AdaVideoRAG**     | Adaptive granularity (coarse/fine windows)       | Hierarchical text, visual, graph indices   | Query-guided scale adaptation with intent classification                               | ❌ No public code yet                 |

