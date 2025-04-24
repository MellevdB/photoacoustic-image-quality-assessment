import os
import cv2
import numpy as np

def load_image_stack_with_ids(folder, file_extension=".png"):
    files = sorted([f for f in os.listdir(folder) if f.endswith(file_extension)])
    stack = []
    image_ids = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img)
            image_ids.append(os.path.join(folder, f))  # Full path or relative
    return (np.stack(stack, axis=0) if stack else np.empty((0, 0, 0))), image_ids