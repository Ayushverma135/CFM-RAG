"""
hash_utils.py
Utility functions for hashing video files and frames for caching purposes.
"""

import hashlib
import numpy as np


def hash_file(filepath, chunk_size=8192):
    """
    Compute SHA256 hash of a file's contents.
    Args:
        filepath (str): Path to the file.
        chunk_size (int): Size of chunks to read at a time.
    Returns:
        str: SHA256 hex digest of the file.
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def hash_frame(frame: np.ndarray) -> str:
    """
    Compute SHA256 hash of a numpy frame (image array).
    Args:
        frame (np.ndarray): Frame data.
    Returns:
        str: SHA256 hex digest of the frame bytes.
    """
    sha256 = hashlib.sha256()
    sha256.update(frame.tobytes())
    return sha256.hexdigest() 