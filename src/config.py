"""
config.py
Defines configuration constants for the Video-RAG pipeline.
"""

import torch
import os


class Config:
    """
    Configuration constants for paths and device selection.
    """
    VIDEO_PATH = "C:/Users/ayush/OneDrive/Desktop/ayush verma/Video-RAG/media/sample-video.mp4"
    CACHE_DIR = "C:/Users/ayush/OneDrive/Desktop/ayush verma/Video-RAG/cache"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"