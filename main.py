import clip.model
import torch
import clip
import  glob
import matplotlib.pyplot as plt
import numpy as np
import cv2 # OpenCV
import seaborn  as sb
import pathlib
from PIL import Image
from ultralytics import YOLO  

import data.val2017 as img_files # image files

IMG_RESIZE = (224, 224)

# Load models
print("### Loading CLIP Model: ViT-L/14 ###")
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU, else CPU
clip_model, preprocess = clip.load("ViT-L/14", device=device) # load model with GPU
#print(clip.available_models())

print("### Loading YOLO Model: YOLOv5n ###")
yolo5 = YOLO("yolov5n.pt")
#print(yolo5.info())
print("=== ### Models Loaded Successfully! ### ===")

# Get a list of img paths from a directory
def get_img_paths(directory):
    paths = []
    for file_path in pathlib.Path(directory).rglob('*'):
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            paths.append(str(file_path))
    return paths

# Functions for CLIP, copied from https://www.youtube.com/watch?v=4LpVRQptdzc by Tech Watt
# Gets the image embeddings
def Images(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(processed_image)
    return image_embeddings

# Gets the text embeddings
def Text(text):
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
    return text_embedding

# Compare image to text captions
def Compare(image, text):
    print(text)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(image, text)  # Compare embeddings similaritiies between image and text
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.ravel(probs)

# Load image
IMG_PATH = "data/COCOval17_200"
#IMG_PATH = "000000002592.jpg"
#img = Image.open('000000002592.jpg')
#print("### Loading Image...")
try:
    print("## Loading image from path...")
    
    # Yolov5 detection
    print("### Detecting with YOLOv5n...")
    yolo_results = yolo5(IMG_PATH)
    # Get detections
    print("### Getting bboxes...")
    boundboxes = yolo_results[0].boxes.xyxy.cpu().numpy() # The x1, y1, x2, y2 bounding box points
except:
    print("--- Could not retrieve image from path and perform detection ! ---")
    exit() # end program

#torch.tensor(IMG_PATH)

#img = img.resize(IMG_RESIZE)

print("### Cropping Images...")
try:
    crops = []
    img_idx = 0
    img_paths = get_img_paths(IMG_PATH) # get image paths
    for box in boundboxes:
        # access each image
        img_bgr = cv2.imread(img_paths[img_idx]) # open a specific image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert to RGB format
        
        # proceed to crop
        x1, y1, x2, y2 = map(int, box) # get bbox coordinates
        # ===== Cropping validation =====
        h, w, _ = img_rgb.shape

        # valid image boundary constraints
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Skip invalid or empty boxes
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid box: {box}")
            continue # go to next image
        # ===============================
        # Safe crop
        cropped = img_rgb[y1:y2, x1:x2] # slice the ndarray image
        crop_pil = Image.fromarray(cropped) # convert to Image object for CLIP preprocess()
        crop_pil.resize(IMG_RESIZE) 
        crops.append(crop_pil) # add to list
        img_idx += 1 # increment index value
except:
    print("--- An error occurred with image cropping ---")
    exit() # end program

"""
boxes = yolo_results[0].boxes
probs = yolo_results[0].probs
print("### Boundboxes ###")
print(boundboxes)
print("### Print(boxes) ###")
print(boxes)
print("### Print(probs) ###")
print(probs)
print("#### YOLO.SHOW ###")
yolo_results[0].show()
print("### End of YOLO results ###")
"""
prompts = ["a gift received", "a wrapped box", "a toy", "a memento", "a birthday present", "a souvenir"] # prompts

print("### Getting Similarities from CLIP...")
for cropped_img in crops:
    similarity = Compare(image=cropped_img, text=prompts)
""""
try:
    for cropped_img in crops:
        similarity = Compare(image=cropped_img, text=prompts)
except:
    print("--- An error occurred ! ---")
    exit() # end program
"""
# For single-image
"""
for i, box in enumerate(boundboxes):
    x1, y1, x2, y2 = map(int, box)
    print(x1, y1, x2, y2)
    cropped = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cropped)
    similarity_result = Compare(image=crop_pil, text=prompts) # using loaded image and text list, still just CLIP, NEED TO CONNECT WITH YOLO
    print(similarity_result)
"""
print("=== ### Results ### ===")
# Visualise results

print("### Getting Cosine Scores...")
"""
cosine_scores = []
for img in crops:  # your object detections as PIL Images
    scores = Compare(img, prompts)
    gift_score = scores[0]
    cosine_scores.append(gift_score)
"""
try:
    cosine_scores = []
    for img in crops:  # your object detections as PIL Images
        scores = Compare(img, prompts)
        print(scores)
        #gift_score = scores[0]
    cosine_scores.append(scores)
except:
    print("--- Could not get scores !! ---")
    exit() # end program

print("### Plotting Histogram...")
plt.figure(figsize=(8,5))
sb.histplot(cosine_scores, bins=10, kde=True, color="skyblue")
plt.title("Cosine Similarity Across All 6 prompts")
plt.xlabel("Cosine Similarity Scores")
plt.ylabel("Frequency")
plt.grid(True)
#plt.tight_layout()
plt.show()
print("### END OF PROGRAM  ###")