import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load Models --------
# Grounding DINO
dino_model_id = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

# SAM
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# -------- Input Image --------
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# -------- Step 1: Grounding DINO Detection --------
text_labels = [["a cat", "a TV remote control"]]
dino_inputs = dino_processor(images=image, text=text_labels, return_tensors="pt").to(device)

with torch.no_grad():
    dino_outputs = dino_model(**dino_inputs)

results = dino_processor.post_process_grounded_object_detection(
    dino_outputs,
    dino_inputs.input_ids,
    box_threshold=0.3,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)[0]

boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]

# -------- Step 2: SAM Segmentation --------
# Convert Grounding DINO boxes to center points
center_points = []
for box in boxes:
    x0, y0, x1, y1 = box
    center_x = ((x0 + x1) / 2).item()
    center_y = ((y0 + y1) / 2).item()
    center_points.append([[center_x, center_y]])

# Only send first point per label to SAM
sam_inputs = sam_processor(image, input_points=[center_points], return_tensors="pt").to(device)

with torch.no_grad():
    sam_outputs = sam_model(**sam_inputs)

masks = sam_processor.image_processor.post_process_masks(
    sam_outputs.pred_masks.cpu(), 
    sam_inputs["original_sizes"].cpu(), 
    sam_inputs["reshaped_input_sizes"].cpu()
)
iou_scores = sam_outputs.iou_scores

# -------- Step 3: Visualize --------
image_np = np.array(image)

for i, mask in enumerate(masks[0]):
    label = labels[i]
    score = scores[i].item()
    iou = torch.max(iou_scores[0][i]).item()
    best_mask = mask.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(best_mask.any(0), alpha=0.5, cmap='jet')  # reduce across channel dimension
    plt.title(f"{label} | DINO Score: {score:.2f}, IoU: {iou:.2f}")
    plt.axis("off")
    plt.show()





#This below code only visualizes the best mask

# import torch
# from PIL import Image
# import requests
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -------- Load Models --------
# # Grounding DINO
# dino_model_id = "IDEA-Research/grounding-dino-tiny"
# dino_processor = AutoProcessor.from_pretrained(dino_model_id)
# dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

# # SAM
# sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
# sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# # -------- Input Image --------
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# # -------- Step 1: Grounding DINO Detection --------
# text_labels = [["a cat", "a TV remote control"]]
# dino_inputs = dino_processor(images=image, text=text_labels, return_tensors="pt").to(device)

# with torch.no_grad():
#     dino_outputs = dino_model(**dino_inputs)

# results = dino_processor.post_process_grounded_object_detection(
#     dino_outputs,
#     dino_inputs.input_ids,
#     box_threshold=0.3,
#     text_threshold=0.25,
#     target_sizes=[image.size[::-1]]
# )[0]

# boxes = results["boxes"]
# scores = results["scores"]
# labels = results["labels"]

# # -------- Step 2: SAM Segmentation --------
# center_points = []
# for box in boxes:
#     x0, y0, x1, y1 = box
#     center_x = ((x0 + x1) / 2).item()
#     center_y = ((y0 + y1) / 2).item()
#     center_points.append([[center_x, center_y]])

# sam_inputs = sam_processor(image, input_points=[center_points], return_tensors="pt").to(device)

# with torch.no_grad():
#     sam_outputs = sam_model(**sam_inputs)

# masks = sam_processor.image_processor.post_process_masks(
#     sam_outputs.pred_masks.cpu(), 
#     sam_inputs["original_sizes"].cpu(), 
#     sam_inputs["reshaped_input_sizes"].cpu()
# )

# iou_scores = sam_outputs.iou_scores[0]  # shape: [num_objects, 3]

# # -------- Step 3: Visualize Only Best Mask --------
# # Get index of the best IoU score across all masks
# best_index = torch.argmax(iou_scores.max(dim=1).values).item()

# best_mask = masks[0][best_index].cpu().numpy()
# label = labels[best_index]
# score = scores[best_index].item()
# iou = torch.max(iou_scores[best_index]).item()

# # Visualize
# image_np = np.array(image)
# plt.figure(figsize=(10, 10))
# plt.imshow(image_np)
# plt.imshow(best_mask.any(0), alpha=0.5, cmap='jet')
# plt.title(f"{label} | DINO Score: {score:.2f}, IoU: {iou:.2f}")
# plt.axis("off")
# plt.show()
