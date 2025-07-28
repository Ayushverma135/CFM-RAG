"""
transcriber.py
Implements the Transcriber class for extracting and caching audio transcriptions from video files.
"""

import subprocess
import numpy as np
import hashlib
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import Config
from cache_utils import CacheUtils


class Transcriber:
    """
    Transcribes audio from video files using Whisper and caches the results.
    """
    def __init__(self):
        """
        Initialize the Whisper processor and model.
        """
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(Config.DEVICE)

    def transcribe(self, video_path: str) -> str:
        """
        Transcribe the audio from a video file, using cache if available.
        Args:
            video_path (str): Path to the video file.
        Returns:
            str: The transcribed text.
        """
        cache_key = hashlib.sha256(video_path.encode()).hexdigest()
        cached = CacheUtils.get(cache_key)
        if cached:
            return cached

        cmd = [
            "ffmpeg", "-i", video_path, "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout.read()
        proc.stdout.close()
        audio = np.frombuffer(raw, dtype=np.float32)
        inputs = self.proc(audio, sampling_rate=16000, return_tensors="pt").to(Config.DEVICE)
        ids = self.model.generate(**inputs)
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        CacheUtils.set(cache_key, text)
        return text