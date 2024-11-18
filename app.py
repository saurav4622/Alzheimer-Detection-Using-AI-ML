import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models

# Loading of trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)  # Initialize the model architecture
    model.fc = torch.nn.Linear(model.fc.in_features, 4) 
    model.load_state_dict(torch.load("alzheimers_cnn_model.pth", map_location=torch.device('cpu')))
    model.eval() 
    return model

# Function for making predictions
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted_class = torch.max(outputs, 1)
    class_labels = ["AD (Alzheimer's Disease)", "CN (Cognitively Normal)", "EMCI (Early Mild Cognitive Impairment)", "LMCI (Late Mild Cognitive Impairment)"]
    return class_labels[predicted_class.item()]

# Streamlit UI
st.title("Alzheimer's Disease Classification")
st.write("Upload an image to predict the category of Alzheimer's disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    model = load_model()

    # Make prediction
    st.write("Classifying...")
    prediction = predict(image, model)
    st.success(f"Prediction: {prediction}")
