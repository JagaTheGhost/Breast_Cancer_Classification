import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Load model
model = tf.keras.models.load_model("Breast_Cancer_Classification.h5")

class_names = ["Benign", "Malignant"]

st.title("🩺 Breast Cancer Classifier")
st.write("Upload a histology image (50x50 px) to check if it's **Benign or Malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((50, 50))
    st.image(img, caption='Uploaded Image', use_column_width=False)

    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    pred_class = class_names[int(prediction[0][0] > 0.5)]
    confidence = float(prediction[0][0])

    st.markdown(f"### 🧠 Prediction: **{pred_class}**")
    st.markdown(f"Confidence Score: **{confidence:.2f}**")
