"""
retriever.py
Implements the Retriever class for querying indexed video and text data using CLIP and LLM.
"""

from groq import Groq
import torch
import numpy as np
import open_clip
from config import Config
import os


class Retriever:
    """
    Retrieves relevant information from indexed video and text data using CLIP embeddings and LLM.
    """
    def __init__(self, texts, text_embs, img_embs, caps, dets, times):
        """
        Initialize the Retriever with embeddings and metadata.
        Args:
            texts: List of text segments.
            text_embs: Text embeddings.
            img_embs: Image embeddings.
            caps: Captions.
            dets: Detections.
            times: Timestamps.
        """
        self.texts = texts
        self.text_embs = text_embs  # shape: [n_texts, 1024]
        self.img_embs = img_embs    # shape: [n_images, 1024]
        self.caps = caps
        self.dets = dets
        self.times = times

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _, self.processor = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

        self.client = Groq(api_key=os.getenv('GROQ_API_KEY', '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'))

    def query(self, q):
        """
        Query the retriever with a user question.
        Args:
            q (str): The user query.
        Returns:
            tuple: (prompt, answer) from the LLM.
        """
        with torch.no_grad():
            q_tokens = self.tokenizer([q]).to(self.device)
            q_emb = self.clip_model.encode_text(q_tokens)
            q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)

        # Normalize corpus embeddings
        corpus_text_embs = torch.tensor(self.text_embs, device=self.device)
        corpus_text_embs = torch.nn.functional.normalize(corpus_text_embs, p=2, dim=-1)

        corpus_img_embs = torch.tensor(self.img_embs, device=self.device)
        corpus_img_embs = torch.nn.functional.normalize(corpus_img_embs, p=2, dim=-1)

        # Cosine similarities
        text_scores = torch.matmul(q_emb, corpus_text_embs.T).squeeze(0)
        img_scores = torch.matmul(q_emb, corpus_img_embs.T).squeeze(0)

        top_text_idx = torch.topk(text_scores, k=3).indices.cpu().numpy()
        top_img_idx = torch.topk(img_scores, k=3).indices.cpu().numpy()

        # Prompt construction
        parts = [f"[Text] {self.texts[i]}" for i in top_text_idx]
        for j in top_img_idx:
            parts.append(f"[Vis {self.times[j]:.2f}s] {self.caps[j]}")
            parts.append("[Det %0.2fs] %s" % (self.times[j], ", ".join(f"{l}({s:.2f})" for l, s in self.dets[j])))

        prompt = "\n".join(parts) + f"\nQ: {q}\nA:"

        # Streaming LLM response from Groq
        resp = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            stream=True
        )

        ans = ""
        for chunk in resp:
            d = chunk.choices[0].delta.content
            if d:
                ans += d
        return prompt, ans