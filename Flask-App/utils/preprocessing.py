from PIL import Image
import numpy as np

def preprocess_image(filepath):
    """
    Preprocess the image for model inference:
    - Open and decode
    - Resize to 224x224
    - Normalize to [0, 1]
    - Apply ImageNet normalization
    """
    img = Image.open(filepath).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, axis=0)
    return arr 