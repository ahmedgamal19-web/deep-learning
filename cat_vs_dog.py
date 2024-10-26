import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# Load your pre-trained model
model = load_model('cat_dog_classifier_model.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match model's input shape
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict the class of the image
def predict_image(img_array):
    prediction = model.predict(img_array)
    class_label = (prediction[0][0] >= 0.5).astype(int)  # 1 for dog, 0 for cat
    return class_label

# Function to evaluate the model (Dummy data for illustration)
def evaluate_model():
    y_true = [0, 1, 0, 1, 1, 0]  # True labels (0: cat, 1: dog)
    y_pred = [0, 1, 1, 0, 1, 0]  # Example predicted labels
    
    report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
    cm = confusion_matrix(y_true, y_pred)

    return report, cm

# Streamlit UI setup
st.title('Cat vs Dog Image Classifier')

# Create tabs for prediction and evaluation
tab1, tab2 = st.tabs(["Predict", "Evaluation"])

# CSS for styling
st.markdown("""
<style>
.header {
    font-size: 24px;
    font-weight: bold;
    color: #fff;
    background-color: #4CAF50;  /* Green */
    padding: 10px;
    border-radius: 5px;
}
.prediction {
    font-size: 64px;
    font-weight: bold;
    color: #fff;
    padding: 20px;
    border-radius: 10px;
    background-color: rgba(33, 150, 243, 0.8); /* Semi-transparent blue */
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 9999;
    text-align: center;  /* Center text */
}
.balloon {
    position: absolute;  /* Fixed position to fill the screen */
    font-size: 50px; /* Size of the emojis */
    z-index: 1000;  /* Ensure emojis are on top */
}
</style>
""", unsafe_allow_html=True)

# JavaScript for balloon falling effect
st.markdown("""
<script>
function fallBalloons(prediction) {
    console.log("Balloons falling for prediction:", prediction);  // Log the prediction to the console
    const numBalloons = 20;  // Number of balloons to fall
    const balloonType = prediction === 0 ? '🐱' : '🐶'; // Cat or dog emoji

    for (let i = 0; i < numBalloons; i++) {
        const balloon = document.createElement('div');
        balloon.className = 'balloon';
        balloon.innerText = balloonType;  // Set the emoji

        // Randomize the starting position of the balloons
        balloon.style.left = Math.random() * 100 + 'vw'; // Random x position
        balloon.style.top = '0'; // Start from the top of the screen
        document.body.appendChild(balloon);

        // Animate falling
        setTimeout(() => {
            balloon.style.transition = 'transform 3s linear';
            balloon.style.transform = 'translateY(100vh)'; // Fall down the screen
        }, 10); // Start animation after a slight delay

        // Remove balloon after falling
        setTimeout(() => {
            balloon.remove();
        }, 3000); // Match this with the animation duration
    }
}
</script>
""", unsafe_allow_html=True)

# Predict tab
with tab1:
    st.header("Upload an Image for Prediction")
    
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            # Process the uploaded image
            img = Image.open(uploaded_file)
            img_array = load_and_preprocess_image(img)

            # Button to predict the class
            if st.button(f"Predict for {uploaded_file.name}"):
                result = predict_image(img_array)

                # Center result display
                if result == 0:
                    st.markdown(f'<div class="prediction">It\'s a cat!</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction">It\'s a dog!</div>', unsafe_allow_html=True)

                # Call the fallBalloons function to trigger the balloon effect
                st.markdown(f"<script>fallBalloons({result});</script>", unsafe_allow_html=True)

# Evaluation tab
with tab2:
    st.header("Model Evaluation")
    
    report, cm = evaluate_model()
    
    # Display classification report
    st.subheader("Classification Report")
    st.text(report)

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    st.write(cm)
