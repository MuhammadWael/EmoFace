# %%
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the pre-trained emotion recognition model
model = load_model('C:/Users/Arabtech/Desktop/EME Borg Notebook/facial_recognition_model.h5')

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Create a Streamlit app
st.title("Emotion Recognition")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image and convert it to grayscale
    image = Image.open(uploaded_image)
    image = image.convert("L")

    # Resize the image to match the model's input size
    image = image.resize((48, 48))

    # Normalize the image
    image = np.array(image) / 255.0

    # Reshape the image to match the input shape expected by the model
    image = np.reshape(image, (1, 48, 48, 1))  # Use 1 channel (grayscale)

    # Make emotion predictions
    output = model.predict(image)
    predicted_emotion = emotion_labels[np.argmax(output)]

    # Display the predicted emotion
    st.subheader("Predicted Emotion:")
    st.write(predicted_emotion)
