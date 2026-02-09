import cv2
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data once
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def preprocess_image(img_bytes, size=(224, 224)):
    """Decodes and preprocesses an image from bytes."""
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    return gray

def load_images_from_df(df):
    """Parses dataframe to return a numpy array of processed images."""
    processed_images = []
    for i in range(len(df)):
        try:
            # Handle string representation of dict if necessary
            img_data = df.loc[i, "image"]
            if isinstance(img_data, str):
                img_dict = ast.literal_eval(img_data)
            else:
                img_dict = img_data
                
            img_bytes = img_dict["bytes"]
            img_processed = preprocess_image(img_bytes)
            processed_images.append(img_processed)
        except Exception as e:
            print(f"Skipped image {i}: {e}")
    return np.array(processed_images)

def clean_text(text):
    """Tokenizes and cleans caption text."""
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return list(set(tokens))