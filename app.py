import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# --- Model download from Google Drive ---
@st.cache_resource
def download_model():
    model_url = "https://drive.google.com/uc?id=1z3SQDnM3qDtQu-5f2dKLwHZfGmjCiEnc"
    model_path = "model.pth"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
    return model_path

# --- Define your model class here ---
class MarineModel(torch.nn.Module):
    def __init__(self):
        super(MarineModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 5)  # Adjust class count
model.load_state_dict(torch.load("resnet18_half_precision.pth", map_location='cpu'))
model = model.half()  # enable half-precision for inference

    def forward(self, x):
        return self.resnet(x)

# --- Load the model ---
def load_model():
    model_path = download_model()
    model = MarineModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model



# --- Image preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# --- Main app ---
def main():
    st.title("🐠 Marine Species Classifier")
    st.write("Upload an image of a marine species, and the model will classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Classifying..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

            # Map class index to names (update this according to your actual classes)
            class_names = ["Shark", "Tuna", "Dolphin", "Whale", "Clownfish"]
            prediction = class_names[predicted.item()]

        st.success(f"Predicted Species: **{prediction}**")

if __name__ == "__main__":
    main()
