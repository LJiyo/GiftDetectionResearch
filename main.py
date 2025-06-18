import torch
import clip
import matplotlib.pyplot as plt
import numpy as np
import cv2 # OpenCV
import seaborn  as sb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import pathlib
from PIL import Image
from ultralytics import YOLO  

# Constants
CM_LABELS = ["Not Gift", "Gift"]
IMG_RESIZE = (224, 224)
IMG_PATH = "data/Gifts_Dataset" # Data source
#IMG_PATH = "000000002592.jpg"
LIST_SIZE = 6
NUM_IMGS = 215
PROMPTS = ["a gift received", "not a gift", "a toy", "a memento", "a birthday present", "a souvenir"] # prompts
TRUE_LABELS = [1, 0, 1, 0, 1, 1] # Ground Truth of what is acceptable as a 'gift' from the prompts
THRESHOLD = 0.6 # acceptance threshold
DIVISION_MATRIX = [NUM_IMGS] * LIST_SIZE # [215]*6 = [215, 215, 215, 215, 215, 215]

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
def image_features(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(processed_image) 
    image_embedding /= image_embedding.norm(dim=1, keepdim=True) # values are normalised     
    return image_embedding

# Gets the text embeddings
def text_features(text):
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens) 
    text_embedding /= text_embedding.norm(dim=1, keepdim=True) # values are normalised 
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

# =====================================
# Load image
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

#img = img.resize(IMG_RESIZE)

# Crop image to bbox size
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

#print("### Getting Similarities from CLIP...")
"""
try:
    text_features = Text(PROMPTS) 
    for cropped_img in crops:
        img_features = Images(cropped_img)

        similarity = Compare(image=cropped_img, text=PROMPTS)
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

# Visualise results
print("=== ### Results ### ===")
print("### Getting Cosine Scores...")

preds_avg = [0] * LIST_SIZE # List to store average totals of all six predictions
# for testing outside try  statement
try:
    cosine_scores = []
    text_features = text_features(PROMPTS)
    for img in crops:  # object detections as PIL Images
        img_features = image_features(img)
        scores = Compare(img, PROMPTS)
        print(scores)

        index = scores.argmax() # index of max score 
        val = scores.max() # max value within scores
        preds_avg[index] += val
        print("preds_avg for: ", index, " is ", preds_avg[index])
    cosine_scores.append(scores) 
    # averages for each prompt
    for preds in preds_avg:
        preds = preds/NUM_IMGS
    print("Preds_avg", preds_avg)
except:
    print("--- Could not get scores !! ---")
    exit() # end program

# ### CONFUSION MATRIX ###
print("### Plotting Confusion Matrix...")
predicted_labels = [1 if score > THRESHOLD else 0 for score in preds_avg]

# Create the confusion matrix
cm = confusion_matrix(TRUE_LABELS, predicted_labels)
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CM_LABELS).plot()

# ### PR CURVE ###
# Compute precision-recall values
precision, recall, thresholds = precision_recall_curve(TRUE_LABELS, preds_avg)
ap_score = average_precision_score(TRUE_LABELS, preds_avg)

plt.figure(figsize=(8,5))
print("### Plotting PR Curve...")
plt.plot(recall, precision, marker='X', label=f'AP = {ap_score:.2f}')
plt.title('Precision–Recall Curve (Gift Detection)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
"""
# ### HISTOGRAM ###
print("### Plotting Histogram...")
plt.title("Cosine Similarity Across All 6 prompts")
sb.histplot(cosine_scores, bins=20, kde=True, color="skyblue")
#plt.hist(cosine_scores, label="6 Prompts")
plt.xlabel("Cosine Similarity Scores")
plt.ylabel("Frequency")
"""
plt.grid(True)
plt.tight_layout()
plt.show()
print("### END OF PROGRAM  ###")
