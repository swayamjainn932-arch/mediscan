import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load pretrained model ONCE
# -------------------------------
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# -------------------------------
# Load NORMAL EYE reference images
# -------------------------------
NORMAL_EYE_FOLDER = r"C:\Users\HP\Downloads\medi\MediScannn\ML-API\eye_disease\Normal_Eyes"

eye_embeddings = []

def get_embedding_from_path(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return model.predict(img, verbose=0)

for file in os.listdir(NORMAL_EYE_FOLDER):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        try:
            emb = get_embedding_from_path(os.path.join(NORMAL_EYE_FOLDER, file))
            eye_embeddings.append(emb)
        except:
            pass

if len(eye_embeddings) == 0:
    raise Exception("No valid eye images found")

# -------------------------------
# Convert BYTES → embedding
# -------------------------------
def get_embedding_from_bytes(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image bytes")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return model.predict(img, verbose=0)

# -------------------------------
# MAIN FUNCTION USED BY FLASK
# -------------------------------
def is_eye_image(image_bytes, threshold=0.60):
    try:
        uploaded_emb = get_embedding_from_bytes(image_bytes)
    except:
        return False, "Image not readable"

    similarities = [
        cosine_similarity(uploaded_emb, ref)[0][0]
        for ref in eye_embeddings
    ]

    score = max(similarities)

    if score >= threshold:
        return True, f"Valid eye image (similarity={score:.2f})"
    else:
        return False, f"Not an eye image (similarity={score:.2f})"
