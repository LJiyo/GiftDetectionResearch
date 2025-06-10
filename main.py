import torch
import clip
import  glob
import numpy as np
from PIL import Image
from ultralytics import YOLO  

# Load models
print("### Loading CLIP Model: ViT-L/14 ###")
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU, else CPU
clip_model, preprocess = clip.load("ViT-L/14", device=device) # load model with GPU
#print(clip.available_models())

print("### Loading YOLO Model: YOLOv5n ###")
yolo5 = YOLO("yolov5n.pt")
print(yolo5.info())
print("Models Loaded Successfully!")


# Functions for CLIP, copied from https://www.youtube.com/watch?v=4LpVRQptdzc by Tech Watt
# Gets the image embeddings
def Images(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(processed_image)
    return image_embeddings

# Gets the text embeddings
def Text(text: str):
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
    return text_embedding

# Compare embeddings similaritiies between image and text
def Compare(image, text: str):
    print(text)
    image_emb = preprocess(image).unsqueeze(0).to(device)
    text_emb = clip.tokenize(text).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.ravel(probs)


# Load image
# image_path = ""
img = Image.open('.jpg')
# image = image.resize((224, 224))

# Yolov5 detection
yolo_results = yolo5(img)
boundboxes = yolo_results[0].boxes.xyxy.cpu().numpy() # The x1, y1, x2, y2 bounding box points

text = ['a gift', 'a wrapped box', 'a toy', 'a bag', 'a birthday present', 'a souvenir'] # prompts


# For each detection: crop + CLIP encoding
text_prompts = clip.tokenize(text).to(device)
image_tensor = Images(img)
""""
# Loop over detected objects
for box in boundboxes:
    x1, y1, x2, y2 = map(int, box)
    object_crop = img.crop((x1, y1, x2, y2))
    object_tensor = preprocess(object_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(object_tensor)
        text_features = clip_model.encode_text(text_prompts)
        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"Detected Object: Gift likelihood = {probs[0][0]:.2f}")

# similarity_result = Compare(image=image, text=text)
# print(similarity_result)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
"""