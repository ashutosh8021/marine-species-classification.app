import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Try to import OpenCV safely ---
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    st.warning("OpenCV (cv2) is not available. Installing lightweight alternative for image processing...")
    OPENCV_AVAILABLE = False
    # Use PIL for resizing instead of cv2
    
# --- Page title ---
st.title("Marine Species Classifier")
st.markdown("Upload an image to classify the marine species and view Grad-CAM heatmap.")

# --- Define class labels ---
classes = ['Species A', 'Species B', 'Species C', 'Species D', 'Species E']  # update as needed

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        state_dict = torch.load("resnet18_half_precision.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model = model.half()
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

# --- Alternative resize function when OpenCV is not available ---
def resize_without_opencv(array, target_size):
    img = Image.fromarray(array)
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)

# --- Alternative colormap function when OpenCV is not available ---
def simple_heatmap(array):
    # Normalize array to 0-255
    array = array - array.min()
    if array.max() > 0:
        array = array / array.max() * 255
    
    # Create a simple red-based heatmap
    heatmap = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = array.astype(np.uint8)  # Red channel
    return heatmap

# --- Grad-CAM helper ---
def generate_gradcam(model, input_tensor, class_idx):
    # Need to enable gradients for input
    input_tensor.requires_grad = True
    
    # Track activations of the last convolutional layer
    target_layer = model.layer4[-1]
    feature_maps = None
    gradients = None
    
    # Hook to capture activations
    def save_features(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    # Hook to capture gradients
    def save_gradients(grad):
        nonlocal gradients
        gradients = grad.detach()
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(save_features)
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Backward pass on target class
    if output.shape[1] > class_idx:  # Ensure class index is valid
        score = output[0, class_idx]
        feature_maps.register_hook(save_gradients)
        score.backward()
        
        # Clean up
        handle_forward.remove()
        input_tensor.requires_grad = False
        
        # Ensure we have both feature maps and gradients
        if feature_maps is not None and gradients is not None:
            # Convert to numpy
            feature_maps = feature_maps[0].float().cpu().numpy()
            gradients = gradients[0].float().cpu().numpy()
            
            # Calculate weights
            weights = np.mean(gradients, axis=(1, 2))
            
            # Apply weights to feature maps and sum
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature_maps[i]
            
            # Apply ReLU
            cam = np.maximum(cam, 0)
            
            # Resize to match input image size
            if OPENCV_AVAILABLE:
                cam = cv2.resize(cam, (224, 224))
                
                # Convert to color heatmap
                cam = np.uint8(255 * cam)
                colored_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                
                # Convert from BGR to RGB
                colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
            else:
                # Use PIL-based alternatives
                cam_resized = resize_without_opencv(cam, (224, 224))
                
                # Normalize for visualization
                if np.max(cam_resized) > 0:
                    cam_resized = cam_resized / np.max(cam_resized)
                
                colored_cam = simple_heatmap(cam_resized * 255)
            
            return colored_cam
    
    # Return empty heatmap if something went wrong
    return np.zeros((224, 224, 3), dtype=np.uint8)

# --- Create overlay function that works with or without OpenCV ---
def create_overlay(original_img, heatmap, alpha=0.6):
    original_array = np.array(original_img.resize((224, 224)))
    
    if OPENCV_AVAILABLE:
        return cv2.addWeighted(original_array, alpha, heatmap, 1-alpha, 0)
    else:
        # Simple alpha blending without OpenCV
        return (alpha * original_array + (1-alpha) * heatmap).astype(np.uint8)

# --- Streamlit Tabs ---
tab1, tab2 = st.tabs(["🔍 Prediction", "🔥 Grad-CAM Visualization"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file and model is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            input_tensor = transform_image(image)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred_idx = torch.argmax(probs).item()
            
            st.markdown(f"### Predicted: **{classes[pred_idx]}**")
            st.markdown("### Probabilities:")
            for i, p in enumerate(probs):
                st.write(f"{classes[i]}: {p.item():.4f}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

with tab2:
    if uploaded_file and model is not None:
        try:
            if 'image' not in locals():
                image = Image.open(uploaded_file).convert("RGB")
            
            st.image(image, caption="Original Image", use_column_width=False)
            
            with st.spinner("Generating Grad-CAM heatmap..."):
                # Need a fresh tensor with gradients enabled for Grad-CAM
                input_tensor = transform_image(image)
                
                # Get prediction index
                with torch.no_grad():
                    output = model(transform_image(image))
                    pred_idx = torch.argmax(torch.nn.functional.softmax(output[0], dim=0)).item()
                
                # Generate Grad-CAM
                cam = generate_gradcam(model, input_tensor, pred_idx)
                
                # Create overlay
                overlay = create_overlay(image, cam, 0.6)
                
                st.image(overlay, caption=f"Grad-CAM Heatmap for {classes[pred_idx]}", use_column_width=True)
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {e}")
            st.error(f"Detailed error: {str(e)}")
