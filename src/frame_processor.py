"""
frame_processor.py
Implements the FrameProcessor class for extracting OCR, captions, detections, and masks from video frames, with parallelization (using processes) and caching.
"""

import numpy as np
from PIL import Image
from config import Config
from concurrent.futures import ProcessPoolExecutor
from cache_utils import CacheUtils
from hash_utils import hash_frame

class FrameProcessor:
    """
    Processes video frames to extract OCR, captions, object detections, and segmentation masks, with parallelization and caching.
    """
    def __init__(self):
        """
        Only BLIP is loaded here for batch captioning. All other models are loaded per process.
        """
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        self.blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl").to(Config.DEVICE)

    @staticmethod
    def _process_single_frame(args):
        """
        Process a single frame for OCR, detection, segmentation, with caching. All model loading is done inside the process.
        Args:
            args: tuple (frame_bytes, shape, dtype_str)
        Returns tuple: (ocr_text, detection, mask)
        """
        import pytesseract
        import torch
        from PIL import Image
        from ultralytics import YOLO
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamProcessor, SamModel
        from cache_utils import CacheUtils
        from hash_utils import hash_frame
        import numpy as np

        frame_bytes, shape, dtype_str = args
        fr = np.frombuffer(frame_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

        # For caching, hash the frame bytes
        frame_hash = hash_frame(fr)
        cache_key = f"frame_{frame_hash}"
        cached = CacheUtils.get(cache_key)
        if cached:
            return cached

        img_pil = Image.fromarray(fr)

        # OCR
        txt = pytesseract.image_to_string(img_pil).strip()
        ocr_text = txt if txt else ""

        # YOLO for generic detection
        yolo = YOLO("yolov8x.pt")
        res = yolo(fr)[0]
        dets = [(yolo.names[int(b.cls)], float(b.conf)) for b in res.boxes]

        # Grounded DINO + SAM (HuggingFace)
        dino_model_id = "IDEA-Research/grounding-dino-tiny"
        dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(Config.DEVICE)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(Config.DEVICE)

        # Grounding DINO prompt from YOLO labels
        prompt = [[label for label, _ in dets]] if dets else [[]]

        # Step 1: Grounding DINO detection
        dino_inputs = dino_processor(
            images=img_pil, text=prompt, return_tensors="pt"
        ).to(Config.DEVICE)
        with torch.no_grad():
            dino_outputs = dino_model(**dino_inputs)
        results = dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.25,
            target_sizes=[img_pil.size[::-1]]
        )[0]

        boxes = results["boxes"]
        labels = results["labels"]
        scores = results["scores"]

        # Step 2: SAM segmentation
        center_points = []
        for box in boxes:
            x0, y0, x1, y1 = box
            center_x = ((x0 + x1) / 2).item()
            center_y = ((y0 + y1) / 2).item()
            center_points.append([[center_x, center_y]])

        if center_points:
            sam_inputs = sam_processor(
                img_pil, input_points=[center_points], return_tensors="pt"
            ).to(Config.DEVICE)
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
            mask_batch = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )
            mask = mask_batch[0]
        else:
            mask = []

        result = (ocr_text, dets, mask)
        CacheUtils.set(cache_key, result)
        return result

    def process(self, frames):
        """
        Process a list of frames to extract OCR, captions, detections, and segmentation masks, with parallelization and caching.
        Args:
            frames (list): List of video frames (numpy arrays).
        Returns:
            tuple: (ocr_texts, captions, detections, masks)
        """
        from PIL import Image
        from transformers import Blip2Processor
        # BLIP captioning (batched, as before)
        imgs = [Image.fromarray(fr) for fr in frames]
        inputs = self.blip_proc(
            images=imgs,
            text=["Describe this image in detail."] * len(imgs),
            return_tensors="pt"
        ).to(Config.DEVICE)
        outputs = self.blip_model.generate(
            **inputs, max_new_tokens=150, num_beams=5,
            length_penalty=1.2, early_stopping=True
        )
        captions = [self.blip_proc.decode(o, skip_special_tokens=True) for o in outputs]

        # Prepare frames as (bytes, shape, dtype) for multiprocessing
        frame_args = [(fr.tobytes(), fr.shape, str(fr.dtype)) for fr in frames]
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._process_single_frame, frame_args))
        ocr_texts, detections, masks = zip(*results)
        return list(ocr_texts), captions, list(detections), list(masks)

