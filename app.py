import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Page title ---
st.title("Marine Species Classifier")
st.markdown("Upload an image to classify the marine species and view Grad-CAM heatmap.")

# --- Define class labels ---
classes = ['Species A', 'Species B', 'Species C', 'Species D', 'Species E']  # update as needed

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state_dict = torch.load("resnet18_half_precision.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.half()
    model.eval()
    return model

model = load_model()

# --- Define transform ---
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).half()

# --- Grad-CAM helper ---
def generate_gradcam(model, input_tensor, class_idx):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    h1 = model.layer4[1].conv2.register_forward_hook(forward_hook)
    h2 = model.layer4[1].conv2.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    output[0, class_idx].backward()

    acts = activations['value'].detach().cpu()[0]
    grads = gradients['value'].detach().cpu()[0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    h1.remove()
    h2.remove()
    return cam

# --- Streamlit Tabs ---
tab1, tab2 = st.tabs(["🔍 Prediction", "🔥 Grad-CAM Visualization"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_tensor = transform_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()

        st.markdown(f"### Predicted: **{classes[pred_idx]}**")
        st.markdown("### Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"{classes[i]}: {p:.4f}")

with tab2:
    if uploaded_file:
        st.image(image, caption="Original Image", use_column_width=False)
        st.write("Generating Grad-CAM heatmap...")

        cam = generate_gradcam(model, input_tensor, pred_idx)
        original = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(original, 0.5, cam, 0.5, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

