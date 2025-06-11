import clip.model
import torch
import clip
import  glob
import numpy as np
import cv2 # OpenCV
import seaborn  as sb
from PIL import Image
from ultralytics import YOLO  

import data.val2017 as img_files # image files

# Load models
print("### Loading CLIP Model: ViT-L/14 ###")
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU, else CPU
clip_model, preprocess = clip.load("ViT-L/14", device=device) # load model with GPU
#print(clip.available_models())

print("### Loading YOLO Model: YOLOv5n ###")
yolo5 = YOLO("yolov5n.pt")
#print(yolo5.info())
print("### Models Loaded Successfully! ###")

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
#IMG_PATH = "data/COCOval17_200"
IMG_PATH = "000000002592.jpg"
#img = Image.open('000000002592.jpg')
#print("### Loading Image...")
try:
    print("## Loading image from path...")
    # Yolov5 detection
    print("### Detecting with YOLOv5n...")
    yolo_results = yolo5(IMG_PATH)
    img_bgr = cv2.imread(IMG_PATH) # open image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert to RGB format
    img_pil = Image.fromarray(img_rgb) # convert from ndarray to Image object
   
except:
    print("--- Could not retrieve image from path and perform detection ! ---")
    exit() # end program

#torch.tensor(IMG_PATH)

#img = img.resize((224, 224))

# Get detections
boundboxes = yolo_results[0].boxes.xyxy.cpu().numpy() # The x1, y1, x2, y2 bounding box points
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
prompts = ["a gift", "a wrapped box", "a toy", "a bag", "a birthday present", "a souvenir"] # prompts
# For each detection: crop + CLIP encoding
#text_embed = Text(text)
#image_embed = Images(img)
#image_input = preprocess(img).unsqueeze(0).to(device)
#text_text = 

print("### Getting Similarities from CLIP...")
# For each YOLO detection
for i, box in enumerate(boundboxes):
    x1, y1, x2, y2 = map(int, box)
    print(x1, y1, x2, y2)
    cropped = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cropped)
    similarity_result = Compare(image=crop_pil, text=prompts) # using loaded image and text list, still just CLIP, NEED TO CONNECT WITH YOLO
    print(similarity_result)

# Visualise results
cosine_scores = []

print("### END OF PROGRAM  ###")


"""
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
"""