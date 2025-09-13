
import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np


model = keras.models.load_model("teeth_model.keras")

class_names = ['CaS', 'OC', 'Gum', 'OT', 'MC', 'OLP', 'CoS'] 


st.set_page_config(page_title="InceptionV3 Classifier", page_icon="🤖")

st.title("🖼️ Teeth Classification 🦷")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    resized_image = image.resize((224, 224))
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)


    st.subheader("🔍 Prediction Results:")

    if len(preds[0]) == len(class_names):
        for i, prob in enumerate(preds[0]):
            st.write(f"{class_names[i]}: {prob:.4f}")
            st.progress(float(prob))

        st.success(f"✅ Predicted Class: {class_names[np.argmax(preds)]}")
    else:
        st.write("Raw Predictions:", preds)
        st.success(f"✅ Predicted Class Index: {np.argmax(preds)}")
