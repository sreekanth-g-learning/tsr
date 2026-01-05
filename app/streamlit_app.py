import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from models.grad_cam import generate_gradcam

model = tf.keras.models.load_model('best_traffic_sign_model.keras')
st.title('Traffic Sign Recognition')
st.subheader("Grad-CAM Comparison Across Models")
uploaded = st.file_uploader('Upload image', type=['jpg','png'])
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
    img = cv2.resize(img, (30,30))
    arr = np.expand_dims(img/255.0, axis=0)
    pred = model.predict(arr)
    st.write('Predicted class:', np.argmax(pred))

    gradcam_img = generate_gradcam(model, arr, np.argmax(pred))
    st.image(gradcam_img, caption='Grad-CAM Visualization', use_column_width=True)
    