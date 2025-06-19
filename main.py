import autolabel as al
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
#IMG_PATH = "000000002592.jpg" # Single image testing
PROMPT_LIST_SIZE = 6
NUM_IMGS = 200
PROMPTS = [
    "a photo of a gift",                # 0
    "a celebratory present",            # 1 
    "a photo of a toy",                 # 2
    "a photo of a grocery item",        # 3 (non-gift)
    "a photo of a birthday present",    # 4 
    "a photo of a tool"]                # 5 (non-gift)
# target classes from objects in "gifts" directory
GIFT_CLASSES = [
    "teddy bear", "teddy bears",
    "cake", "cakes"
    "donut", "donuts"
    "person", # singular in case they are holding a gift object + account forr dolls
    "potted plant", "plant"
    "flower"
]
GIFT_IDX = [0, 1, 2, 4]
NON_GIFT_IDX = [3, 5]
TRUE_PROMPTS = [1, 1, 1, 0, 1, 0] # Ground Truth of what is acceptable as a 'gift' from the prompts
TRUE_LABELS = al.true_labels # Grouund truth labels
# Uncomment below to check labels output
"""
print("True Labels", TRUE_LABELS)
exit()
"""
THRESHOLD = 0.6 # acceptance threshold


# Load models
print("### Loading CLIP Model: ViT-L/14 ###")
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU, else CPU
clip_model, preprocess = clip.load("ViT-L/14", device=device) # load model with GPU
#print(clip.available_models())

print("### Loading YOLO Model: YOLOv5n ###")
yolo5 = YOLO("yolov5n.pt")
# get target classes ids 
gift_class_ids = [k for k, v in yolo5.names.items() if v in GIFT_CLASSES]
#print(yolo5.info()) # Info about the model
print("=== ### Models Loaded Successfully! ### ===")

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
    #print(text)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(image, text)  # Compare embeddings similaritiies between image and text
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.ravel(probs)

# =====================================
# Load image and detect with YOLO
try:
    print("## Loading image from path...")
    # Yolov5 detection
    print("### Detecting with YOLOv5n...")
    img_paths = al.get_img_paths(IMG_PATH) # get image paths
    # Get detections
    yolo_results = yolo5(img_paths)
    print("=== END OF DETECTIONS ===")
except:
    print("--- Could not retrieve image from path and perform detection ! ---")
    exit() # end program

"""
# Get boundingboxes from detections
try:
    crops = []
    print("### Getting bboxes...")
    boundboxes = yolo_results[0].boxes.xyxy.cpu().numpy() # The x1, y1, x2, y2 bounding box points of the first detection only
    # access each image from results
    for img_idx, result in enumerate(yolo_results):
        img_bgr = cv2.imread(img_paths[img_idx]) # open a specific image
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert to RGB format
        h, w, _ = img_rgb.shape

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            class_id = classes[i]
            # uncommenting this gives 98 detections accepted
            if class_id not in gift_class_ids:
                continue # skip non-gift classes
            
            # Crop image to bbox size
            x1, y1, x2, y2 = map(int, box) # get bbox coordinates
            # ===== Cropping validation =====
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
    print("Crops size: ", len(crops))



except:
    print("--- Could not retrieve boundingboxes ! ---")
    print("--- An error occurred with image cropping ! ---")
    exit()

#print("### Getting Similarities from CLIP...")

try:
    text_features = Text(PROMPTS) 
    for cropped_img in crops:
        img_features = Images(cropped_img)

        similarity = Compare(image=cropped_img, text=PROMPTS)
except:
    print("--- An error occurred ! ---")
    exit() # end program

# For single-image

for i, box in enumerate(boundboxes):
    x1, y1, x2, y2 = map(int, box)
    print(x1, y1, x2, y2)
    cropped = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cropped)
    similarity_result = Compare(image=crop_pil, text=prompts) # using loaded image and text list, still just CLIP, NEED TO CONNECT WITH YOLO
    print(similarity_result)


# Visualise results
print("======== ### Results ### ========")
print("### Getting Cosine Scores...")
cosine_scores = np.zeros(PROMPT_LIST_SIZE) # List for totalling prompt cosine scores
preds_avg = np.zeros(PROMPT_LIST_SIZE) # List to store average totals of all six predictions
try:
    # Lists for confusion matrix
    label_preds = [] 
    prompt_idx_preds = []
    #text_features = text_features(PROMPTS) 
    for img in crops:  # object detections as PIL Images
        #img_features = image_features(img)
        score = Compare(img, PROMPTS) # outputs a (6,) shape of scores for all 6 prompts
        
        # track index of highest score within the output
        prompt_idx_preds.append(int(score.argmax()))

        # grouping into gift and non-gift classes
        gift_score = max([score[i] for i in GIFT_IDX])
        #print("Gift Score: ", gift_score)
        non_gift_score = max([score[i] for i in NON_GIFT_IDX])
        #print("Non-gift score: ", non_gift_score)

        # predicting final label
        label = 1 if gift_score > non_gift_score else 0
        #print("Label = ", label)
        label_preds.append(label) # list for tracking gift or non-gift for each image
        #print("label_preds: ", label_preds)
        cosine_scores += score 
        print(score)

        
        index = score.argmax() # index of max score 
        val = score.max() # max value within scores
        preds_avg[index] += val  # iterative total
        print("preds_avg for: ", index, " is ", preds_avg[index])
        
    # averages for each prompt score
    prompt_avgs = cosine_scores / len(crops) 
    print("====================================")
    print("Prompt averages: ", prompt_avgs)
    print("label_preds size: ", len(label_preds))
except:
    print("--- Could not get scores !! ---")
    exit() # end program
"""
try:
    print("### Processing Images with Fallback for Missing Crops...")

    # Required for CLIP scoring and confusion matrix
    label_preds = []
    prompt_idx_preds = []
    cosine_scores = np.zeros(PROMPT_LIST_SIZE)

    for img_idx, result in enumerate(yolo_results):
        img_bgr = cv2.imread(img_paths[img_idx])
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

        if len(boxes) == 0:
            # Fallback to full image
            full_image = Image.fromarray(img_rgb).resize(IMG_RESIZE)
            scores = Compare(full_image, PROMPTS)
        else:
            # Use first valid crop only 
            x1, y1, x2, y2 = map(int, boxes[0])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid box: {boxes[0]}")
                continue

            cropped = img_rgb[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cropped).resize(IMG_RESIZE)
            scores = Compare(crop_pil, PROMPTS)

        # Record prompt index
        prompt_idx_preds.append(int(scores.argmax()))

        # Score grouping
        gift_score = max([scores[i] for i in GIFT_IDX])
        non_gift_score = max([scores[i] for i in NON_GIFT_IDX])

        label = 1 if gift_score > non_gift_score else 0
        label_preds.append(label)
        cosine_scores += scores
except:
    print("--- An Error Occurred With Getting Images ! ---")

# ### CONFUSION MATRIX ###
print("### Plotting Confusion Matrix...")
#predicted_labels = [1 if score > THRESHOLD else 0 for score in preds_avg]

# Create the confusion matrix
cm = confusion_matrix(TRUE_LABELS, label_preds)
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CM_LABELS).plot()

# ### PR CURVE ###
# Compute precision-recall values
precision, recall, thresholds = precision_recall_curve(TRUE_LABELS, label_preds)
ap_score = average_precision_score(TRUE_LABELS, label_preds)

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
