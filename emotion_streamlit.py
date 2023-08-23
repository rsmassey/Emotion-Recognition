import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('emotion_model.h5')  # Use tf.keras.models instead of keras.models
    return model

model = load_model()

st.title("Emotion Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def predict(image):
    # Convert image to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 48x48 pixels
    image = image.resize((48, 48))
    # Convert to numpy array and normalize
    image = np.asarray(image) / 255.0
    # Ensure the shape is (1, 48, 48, 1)
    image = np.expand_dims(image, axis=[0, -1])
    # Predict
    predictions = model.predict(image)
    return predictions

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    image = Image.open(uploaded_file)
    prediction = predict(image)

    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    st.write(f"Predicted Emotion: {emotion_classes[np.argmax(prediction)]}")
