"""
cache_utils.py
Implements the CacheUtils class for caching and retrieving objects using pickle files.
"""

import os
import pickle
from config import Config


class CacheUtils:
    """
    Utility class for caching and retrieving objects using pickle files in the cache directory.
    """
    @staticmethod
    def get(key: str):
        """
        Retrieve an object from cache by key.
        Args:
            key (str): The cache key.
        Returns:
            The cached object if it exists, else None.
        """
        path = os.path.join(Config.CACHE_DIR, f"{key}.pkl")
        return pickle.load(open(path, 'rb')) if os.path.exists(path) else None

    @staticmethod
    def set(key: str, obj):
        """
        Store an object in cache with the given key.
        Args:
            key (str): The cache key.
            obj: The object to cache.
        """
        path = os.path.join(Config.CACHE_DIR, f"{key}.pkl")
        pickle.dump(obj, open(path, 'wb'))