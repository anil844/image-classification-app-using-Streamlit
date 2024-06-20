import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Load the pre-trained model
model = load_model('C:\\Users\\hp\\Desktop\\upload github data\\cifar10_model.h5')

# Define the class labels
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(img, model):
    # Convert the image to RGB (if not already)
    img = img.convert('RGB')
    # Preprocess the image
    img = img.resize((32, 32))  # Resize image to 32x32 as expected by the model
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app layout
st.title('Image Classification with CNN')
st.write('Upload an image of a CIFAR-10 object for classification.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict and display the result
    if st.button('Classify'):
        prediction = predict_image(img, model)
        st.write(f'Prediction: {prediction}')
