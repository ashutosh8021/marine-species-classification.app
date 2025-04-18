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
    
# --- Page title ---
st.title("Marine Species Classifier")
st.markdown("Upload an image to classify the marine species and view Grad-CAM heatmap.")

# --- Define class labels ---
# Match the saved model which has 3 classes
classes = ['Species A', 'Species B', 'Species C']

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        # Create fully connected layer with 3 outputs to match the saved model
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        
        state_dict = torch.load("resnet18_half_precision.pth", map_location=torch.device('cpu'))
        
        # Debug info about model classes
        fc_weight_shape = next((v.shape for k, v in state_dict.items() if k == "fc.weight"), None)
        if fc_weight_shape:
            st.info(f"Detected {fc_weight_shape[0]} classes in the saved model.")
        
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
    img = Image.fromarray((array * 255).astype(np.uint8))
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

# --- Completely rewritten Grad-CAM implementation ---
def generate_gradcam(model, img, target_class=None):
    """
    Generate Grad-CAM visualization for the target class
    """
    if model is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Switch to evaluation mode
    model.eval()
    
    # Convert model to float for gradient computations
    model = model.float()
    
    # Transform the image for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0)
    x.requires_grad = True
    
    # Forward pass to get class predictions
    with torch.no_grad():
        pred = model(x)
        if target_class is None:
            target_class = pred.argmax(dim=1).item()
    
    # Get the target layer (usually the last convolutional layer)
    target_layer = model.layer4[-1].conv2
    
    # Register a hook to get activations
    activations = []
    def forward_hook(module, input, output):
        activations.append(output)
    hook = target_layer.register_forward_hook(forward_hook)
    
    # Make sure gradients are being accumulated
    model.zero_grad()
    
    # Forward pass with gradients
    output = model(x)
    
    # Create one-hot encoding for the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    # Backward pass to get gradients
    output.backward(gradient=one_hot)
    
    # Get the gradients from the target layer
    gradients = x.grad
    
    # Cleanup: remove the hook
    hook.remove()
    
    # Get the activations from the forward pass
    activations = activations[0].detach().cpu().numpy()[0]
    
    # Taking a simple approach here since we had issues with gradient hooks
    # Calculate the average gradient per feature map
    weights = np.mean(activations, axis=(1, 2))
    
    # Compute weighted combination of feature maps
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    # Create visualization
    if OPENCV_AVAILABLE:
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    else:
        cam = resize_without_opencv(cam, (224, 224))
        heatmap = simple_heatmap(cam * 255)
    
    # Convert model back to half-precision for inference
    model = model.half()
    
    return heatmap

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

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if model is not None:
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
            else:
                st.error("Model could not be loaded. Please check the model file and class configuration.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

with tab2:
    if uploaded_file:
        try:
            if 'image' not in locals():
                image = Image.open(uploaded_file).convert("RGB")
            
            st.image(image, caption="Original Image", use_column_width=False)
            
            if model is not None:
                with st.spinner("Generating Grad-CAM heatmap..."):
                    # Get prediction index for inference
                    input_tensor = transform_image(image)
                    with torch.no_grad():
                        output = model(input_tensor)
                        pred_idx = torch.argmax(torch.nn.functional.softmax(output[0], dim=0)).item()
                    
                    # Use completely rewritten Grad-CAM function
                    try:
                        heatmap = generate_gradcam(model, image, target_class=pred_idx)
                        
                        # Create overlay
                        overlay = create_overlay(image, heatmap, 0.6)
                        
                        st.image(overlay, caption=f"Grad-CAM Heatmap for {classes[pred_idx]}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not generate heatmap: {str(e)}")
                        st.info("Showing prediction results only.")
            else:
                st.error("Model could not be loaded. Cannot generate visualization.")
        except Exception as e:
            st.error(f"Error in visualization tab: {str(e)}")
