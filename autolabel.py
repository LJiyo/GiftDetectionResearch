import os
import pathlib

DATASET_GIFT_PATH = "data/Gifts_Dataset/gifts"
DATASET_NON_GIFT_PATH = "data/Gifts_Dataset/non_gifts"
DATASET_PARENT_PATH = "data/Gifts_Dataset"

true_labels = []

def get_img_paths(directory):
    paths = []
    for file_path in pathlib.Path(directory).rglob('*'):
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            paths.append(str(file_path))
    return paths

image_path1 = get_img_paths(DATASET_GIFT_PATH)
image_path2 = get_img_paths(DATASET_NON_GIFT_PATH)
image_path3 = get_img_paths(DATASET_PARENT_PATH)

"""  
# Label all gifts as 1
for path in image_path1:
    if "gifts" in path:
        true_labels.append(1)
    else:
        print("Error Labelling for: " + path)
        exit()

# Label all non-gifts as 0
for path in image_path2:
    if "non_gifts" in path:
        true_labels.append(0)
    else:
        print("Error Labelling for: " + path)
        exit()
"""
# Label gifts as 1 and non_gifts as 0
for path in image_path3:
    if "non_gifts" in path:
        true_labels.append(0)
    elif "gifts" in path:
        true_labels.append(1)
    else:
        print("Error Labelling for: " + path)
        exit()
      
print(true_labels)