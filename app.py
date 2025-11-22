import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

# PAGE CONFIGURATION 
st.set_page_config(
    page_title="Aerial Surveillance System",
    layout="wide"
)

# SETUP DEVICE & CLASSES
# Use CPU for deployment to avoid CUDA errors on non-GPU machines
DEVICE = torch.device('cpu') 
CLASSES = ['Bird', 'Drone']

# DEFINE CLASSIFICATION MODEL ARCHITECTURE
def get_classification_model():
    # Load empty ResNet18
    model = models.resnet18(weights=None)
    
    # Re-create the custom head (Must match training exactly!)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

# CACHED MODEL LOADING (The "Anti-Crash" Logic)
# We use @st.cache_resource so models load ONLY ONCE, not every refresh

@st.cache_resource
def load_classifier(path='best_transfer_model.pth'):
    try:
        model = get_classification_model()
        # map_location='cpu' ensures it runs even if trained on GPU
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"File not found: {path}. Please move your .pth file to the app folder.")
        return None
    except Exception as e:
        st.error(f"Error loading Classifier: {e}")
        return None

@st.cache_resource
def load_yolo(path='runs/detect/drone_bird_yolo/weights/best.pt'):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        # Fallback to generic model if custom one is missing (prevents crash)
        st.warning(f"Custom YOLO not found at {path}. Loading standard YOLOv8n for demo.")
        return YOLO('yolov8n.pt')

# PREPROCESSING FUNCTION
def process_image(image):
    # Same transforms as Validation Phase
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# MAIN UI LAYOUT
st.title("Aerial Object Surveillance System")
st.markdown("### AI-Powered Bird vs. Drone Detection")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Mode:", ["Classification (What is it?)", "Object Detection (Where is it?)"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

# File Uploader
uploaded_file = st.file_uploader("Upload an Aerial Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display Image
    col1, col2 = st.columns(2)
    
    # Open image with PIL
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image, caption="Uploaded Image", width='stretch')

    # LOGIC BRANCHING
    if mode == "Classification (What is it?)":
        st.sidebar.info("Model: ResNet18 (Transfer Learning)")
        
        # Load Model
        classifier = load_classifier()
        
        if classifier:
            # Inference
            with torch.no_grad():
                img_tensor = process_image(image).to(DEVICE)
                output = classifier(img_tensor).item()
                
            # Logic: Output is 0-1. >0.5 is Drone, <0.5 is Bird
            label = "Drone" if output > 0.5 else "Bird"
            prob = output if output > 0.5 else 1 - output
            
            # Display Result
            with col2:
                st.markdown(f"### Prediction: **{label}**")
                st.metric(label="Confidence Score", value=f"{prob*100:.2f}%")
                
                # Dynamic Progress Bar
                if label == "Drone":
                    st.progress(int(prob * 100), text="Drone Threat Level")
                else:
                    st.progress(int(prob * 100), text="Biological Certainty")

    elif mode == "Object Detection (Where is it?)":
        st.sidebar.info("Model: YOLOv8 (Custom Trained)")
        
        # Load YOLO
        detector = load_yolo()
        
        if detector:
            # Inference
            results = detector.predict(image, conf=confidence_threshold, iou=0.5)
            
            # Plot Results
            # YOLO plots return a numpy array in BGR, we convert to RGB
            res_plotted = results[0].plot()
            res_rgb = res_plotted[:, :, ::-1] 
            
            with col2:
                st.image(res_rgb, caption="Detected Objects", width='stretch')

                
                # Show detections text
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"Found {len(boxes)} object(s)!")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = results[0].names[cls_id]
                        st.write(f"**{name}** - {conf:.2f} confidence")
                else:
                    st.warning("No objects detected. Try lowering the confidence threshold.")

else:
    st.info("Please upload an image to begin analysis.")