import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
# Title
st.title("Marine Species Classifier")

# Load your model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)  # Change 8 to your actual number of classes
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names (update these to your actual marine species)
classes = ['Species A', 'Species B', 'Species C', 'Species D', 'Species E', 'Species F', 'Species G', 'Species H']

# Image preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Upload image
uploaded_file = st.file_uploader("Upload an image of a marine species", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Transform & Predict
    input_tensor = transform_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = classes[torch.argmax(probabilities)]

    # Output
    st.write("### Prediction:")
    st.write(f"**{predicted_class}**")
    st.write("### Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"{classes[i]}: {prob:.4f}")
