"""
scene_extractor.py
Implements the SceneExtractor class for extracting scenes and key frames from a video, with caching support.
"""

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from decord import VideoReader, cpu
from cache_utils import CacheUtils
from hash_utils import hash_file


class SceneExtractor:
    """
    Extracts scenes and key frames from a video file, with caching.
    """
    def extract(self, video_path: str):
        """
        Extract scenes and key frames from the given video path, using cache if available.
        Args:
            video_path (str): Path to the video file.
        Returns:
            tuple: (frames, times) where frames is a list of key frames and times is a list of timestamps.
        """
        video_hash = hash_file(video_path)
        cache_key = f"scene_{video_hash}"
        cached = CacheUtils.get(cache_key)
        if cached:
            return cached

        vid_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=5.0))
        vid_manager.start()
        scene_manager.detect_scenes(frame_source=vid_manager)
        scenes = scene_manager.get_scene_list()
        vr = VideoReader(video_path, ctx=cpu())
        frames, times = [], []
        for (start, _) in scenes:
            frame_index = start.get_frames()
            frame = vr.get_batch([frame_index]).asnumpy()[0]
            frames.append(frame)
            times.append(start.get_seconds())
        result = (frames, times)
        CacheUtils.set(cache_key, result)
        return result