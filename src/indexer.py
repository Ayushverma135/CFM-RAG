"""
indexer.py
Implements the Indexer class for building text and image embeddings from video data, with per-item caching.
"""

import numpy as np
from PIL import Image
import open_clip
import torch
from config import Config
from cache_utils import CacheUtils
from hash_utils import hash_frame
import hashlib


def hash_text(text: str) -> str:
    """
    Compute SHA256 hash of a text string.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class Indexer:
    """
    Builds text and image embeddings for video frames and associated text, with per-item caching.
    """
    def __init__(self):
        """
        Initialize the Indexer with CLIP model and tokenizer.
        """
        self.clip_model, _, self.clip_proc = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(Config.DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def build(self, asr, ocr, caps, frames):
        """
        Build text and image embeddings from ASR, OCR, captions, and frames, with per-item caching.
        Args:
            asr: Automatic speech recognition text.
            ocr: List of OCR texts.
            caps: List of captions.
            frames: List of video frames (as numpy arrays).
        Returns:
            tuple: (texts, text_embeddings, image_embeddings)
        """
        # Combine and clean texts
        texts = [t for t in ([asr] + ocr + caps) if isinstance(t, str) and t.strip()]

        # TEXT Embeddings using CLIP, with caching
        txt_embs = []
        for t in texts:
            t_hash = hash_text(t)
            cache_key = f"text_emb_{t_hash}"
            cached = CacheUtils.get(cache_key)
            if cached is not None:
                txt_embs.append(cached)
            else:
                with torch.no_grad():
                    txt_tokens = self.tokenizer([t]).to(Config.DEVICE)
                    emb = self.clip_model.encode_text(txt_tokens)[0].cpu().numpy().astype("float32")
                CacheUtils.set(cache_key, emb)
                txt_embs.append(emb)
        txt_embs = np.stack(txt_embs)
        print("Text embedding shape:", txt_embs.shape)

        # IMAGE Embeddings using CLIP, with caching
        img_embs = []
        for fr in frames:
            f_hash = hash_frame(fr)
            cache_key = f"img_emb_{f_hash}"
            cached = CacheUtils.get(cache_key)
            if cached is not None:
                img_embs.append(cached)
            else:
                inp = self.clip_proc(Image.fromarray(fr)).unsqueeze(0).to(Config.DEVICE)
                with torch.no_grad():
                    emb = self.clip_model.encode_image(inp)[0].cpu().numpy().astype("float32")
                CacheUtils.set(cache_key, emb)
                img_embs.append(emb)
        img_embs = np.stack(img_embs)
        print("Image embedding shape:", img_embs.shape)

        return texts, txt_embs, img_embs