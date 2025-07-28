"""
main.py
Entry point for running the VideoRAGPipeline on a sample video and query.
"""

from pipeline import VideoRAGPipeline
from config import Config


if __name__ == "__main__":
    pipeline = VideoRAGPipeline(Config.VIDEO_PATH)
    prompt, answer = pipeline.run("How many tin cans are there in the video?")
    print("Prompt:\n", prompt)
    print("Answer:\n", answer)