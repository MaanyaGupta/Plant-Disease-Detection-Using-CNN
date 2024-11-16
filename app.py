import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load model and class indices
model = tf.keras.models.load_model("model_1.h5")  # Adjust path if needed
class_indices = json.load(open("class_indices (1).json"))

# Helper functions
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # Ensure this file exists in the correct directory
    try:
        st.image(image_path, use_container_width=True)
    except Exception as e:
        st.error("Error loading home page image. Ensure 'home_page.jpeg' is in the correct directory.")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "Disease Recognition":
    st.title('Plant Disease Classifier')
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
