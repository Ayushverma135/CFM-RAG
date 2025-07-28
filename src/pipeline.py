"""
pipeline.py
Defines the VideoRAGPipeline class for orchestrating the video retrieval-augmented generation pipeline.
"""

import os
from config import Config
from transcriber import Transcriber
from scene_extractor import SceneExtractor
from frame_processor import FrameProcessor
from indexer import Indexer
from retriever import Retriever


class VideoRAGPipeline:
    """
    Orchestrates the video RAG pipeline: scene extraction, transcription, frame processing, indexing, and retrieval.
    """
    def __init__(self, video_path: str):
        """
        Initialize the pipeline with the given video path.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        self.video_path = video_path
        self.transcriber = Transcriber()
        self.extractor = SceneExtractor()
        self.processor = FrameProcessor()
        self.indexer = Indexer()
        self.retriever = None

    def run(self, query: str):
        """
        Run the full pipeline for a given query.
        Args:
            query (str): The user query to retrieve information for.
        Returns:
            The result from the retriever.
        """
        print("Extracting scenes...")
        frames, times = self.extractor.extract(self.video_path)

        print("Transcribing audio...")
        asr = self.transcriber.transcribe(self.video_path)

        print("Processing frames...")
        ocr, caps, dets, masks = self.processor.process(frames)

        print("Building index...")
        texts, t_idx, i_idx = self.indexer.build(asr, ocr, caps, frames)

        print("Initializing retriever...")
        self.retriever = Retriever(texts, t_idx, i_idx, caps, dets, times)

        print("Querying retriever...")
        result = self.retriever.query(query)

        print("Query complete.")
        return result