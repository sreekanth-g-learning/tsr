import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import sys
import os

# 🔧 Fix imports for Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.grad_cam import make_gradcam_heatmap
from utils.visualization import get_superimposed_image

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 30
MODEL_PATH = "checkpoints/grid_33/best.keras"
LAST_CONV_LAYER = "conv2d_12"

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing for vehicles >3.5t",
    "Right-of-way at intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles >3.5t prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing >3.5t"
]

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")
st.title("🚦 Traffic Sign Recognition with Grad-CAM")

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Uploaded Image")
    st.image(image, width=250)

    # Preprocess
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0)

    # Prediction
    #preds = model.predict(input_tensor)
    #class_idx = np.argmax(preds[0])
    #confidence = preds[0][class_idx] * 100

    #st.subheader("🧠 Prediction")
    #st.write(f"**Predicted Class:** {CLASS_NAMES[class_idx]}")
    #st.write(f"**Confidence:** {confidence:.2f}%")

    preds = model.predict(input_tensor)[0]

    # Get Top-5 indices
    top5_idx = np.argsort(preds)[::-1][:5]
    top5_scores = preds[top5_idx] * 100

    st.subheader("🧠 Prediction")

    st.write(
        f"**Top Prediction:** {CLASS_NAMES[top5_idx[0]]} "
        f"({top5_scores[0]:.2f}%)"
    )

    MIN_CONFIDENCE = 0.5  # %
    # Filter out zero-confidence predictions
    filtered_rows = [
        (rank + 1, CLASS_NAMES[idx], score)
        for rank, (idx, score) in enumerate(zip(top5_idx, top5_scores))
        if score >= MIN_CONFIDENCE
    ]

    if filtered_rows:
        top5_data = {
            "Rank": [r[0] for r in filtered_rows],
            "Class": [r[1] for r in filtered_rows],
            "Confidence (%)": [f"{r[2]:.2f}" for r in filtered_rows]
        }

        st.table(top5_data)
    else:
        st.info("No confident predictions above 0%.")

    # Grad-CAM
    for layer in model.layers:
        if layer.name.startswith("conv2d_"):
            last_conv_layer = layer.name  

    if last_conv_layer is None:
        raise ValueError(f"No Conv2D layer found in model")             
        
        print(f"Using last conv layer: {last_conv_layer} for model: {name}")
    heatmap,pred_clas = make_gradcam_heatmap(           
            img_array= input_tensor,
             model = model,  
             last_conv_layer_name=last_conv_layer
        )

    overlay = get_superimposed_image(img_array, heatmap)

    st.subheader("🔥 Grad-CAM Visualization")
    st.image(overlay, width=300)
