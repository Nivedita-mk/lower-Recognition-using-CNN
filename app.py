import os
import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import streamlit as st

# Header for the app
st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
model = load_model('Flower_Recog_Model.h5')

# Function to classify images
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100) + '%'
    return outcome

# File uploader in Streamlit
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Save the uploaded file
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, width=200)

    # Classify the uploaded image
    st.markdown(classify_images(os.path.join('upload', uploaded_file.name)))
