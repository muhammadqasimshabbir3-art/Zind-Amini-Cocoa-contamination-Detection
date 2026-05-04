"""
Zind Amini Cocoa Contamination Detection - Hugging Face Spaces Application
This app provides a web interface for detecting cocoa plant diseases using YOLOv8
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# ==================== CONFIGURATION ====================
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.5
MAX_IMAGE_SIZE = 1280
DEVICE = "cpu"  # Using CPU for inference (more stable and widely compatible)

# Class mapping (update based on your dataset)
CLASS_NAMES = {
    0: "Healthy",
    1: "Diseased",
    2: "Contaminated"
    # Add more classes as per your model
}


# ==================== HELPER FUNCTIONS ====================

def load_model(model_path: str = MODEL_PATH):
    """Load YOLOv8 model"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(model_path)
        model.to(DEVICE)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def process_image(image: np.ndarray, model: YOLO, conf: float) -> Tuple[Image.Image, Dict]:
    """
    Process image and detect cocoa diseases
    
    Args:
        image: Input image as numpy array
        model: YOLOv8 model
        conf: Confidence threshold
        
    Returns:
        Tuple of (annotated image, detection results)
    """
    try:
        # Perform inference
        results = model.predict(image, conf=conf, verbose=False, device=DEVICE)
        
        # Convert to PIL Image for drawing
        annotated_image = results[0].plot()
        pil_image = Image.fromarray(annotated_image)
        
        # Parse detections
        detection_info = []
        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for idx, (box, conf_val, class_idx) in enumerate(zip(boxes, confs, classes)):
                x1, y1, x2, y2 = box
                class_name = results[0].names.get(int(class_idx), "Unknown")
                
                detection_info.append({
                    "Detection": idx + 1,
                    "Class": class_name,
                    "Confidence": f"{conf_val:.4f}",
                    "Coordinates": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                })
        
        return pil_image, detection_info
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def detect_disease(image_input, confidence_slider):
    """
    Main Gradio interface function for disease detection
    
    Args:
        image_input: Input image from Gradio
        confidence_slider: Confidence threshold
        
    Returns:
        Annotated image and detection results
    """
    try:
        # Load model
        model = load_model(MODEL_PATH)
        
        # Convert PIL image to numpy array if needed
        if isinstance(image_input, Image.Image):
            image_np = np.array(image_input)
        else:
            image_np = image_input
        
        # Process image
        annotated_image, detections = process_image(image_np, model, confidence_slider)
        
        # Format output message
        if detections:
            output_text = f"✅ Detections Found: {len(detections)}\n\n"
            for det in detections:
                output_text += f"• {det['Detection']}. {det['Class']} (Conf: {det['Confidence']})\n"
                output_text += f"  Coords: {det['Coordinates']}\n"
        else:
            output_text = "No diseases detected! ✨ Plant appears healthy."
        
        return annotated_image, output_text
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def batch_process_images(image_list: List, confidence: float) -> List[Dict]:
    """
    Process multiple images for batch inference
    
    Args:
        image_list: List of images
        confidence: Confidence threshold
        
    Returns:
        List of detection results
    """
    try:
        model = load_model(MODEL_PATH)
        results_list = []
        
        for idx, image in enumerate(image_list):
            annotated, detections = process_image(np.array(image), model, confidence)
            results_list.append({
                "image_idx": idx,
                "annotated_image": annotated,
                "detections": detections
            })
        
        return results_list
        
    except Exception as e:
        raise Exception(f"Batch processing error: {str(e)}")


# ==================== GRADIO INTERFACE ====================

def create_interface():
    """Create Gradio interface for the application"""
    
    with gr.Blocks(title="Cocoa Disease Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌱 Zind Amini Cocoa Contamination Detection
        
        ### Detect cocoa plant diseases using YOLOv8 Object Detection
        
        This application uses advanced computer vision to identify diseases and contamination 
        in cocoa plants. Upload an image of a cocoa plant to get started!
        
        **Features:**
        - Real-time disease detection
        - Adjustable confidence threshold
        - Detailed detection information
        - GPU-accelerated inference
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input")
                image_input = gr.Image(
                    label="Upload Cocoa Plant Image",
                    type="pil",
                    interactive=True
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Higher values = stricter detection"
                )
                
                detect_button = gr.Button(
                    "🔍 Detect Disease",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                annotated_image = gr.Image(
                    label="Annotated Image",
                    type="pil",
                    interactive=False
                )
                
                detection_output = gr.Textbox(
                    label="Detection Results",
                    lines=10,
                    interactive=False,
                    value="Upload an image and click 'Detect Disease' to begin"
                )
        
        # Examples
        gr.Examples(
            examples=[
                # Add example images if available
            ],
            inputs=[image_input, confidence_slider],
            outputs=[annotated_image, detection_output],
            fn=detect_disease,
            cache_examples=True,
        )
        
        # Connect button click to detection function
        detect_button.click(
            fn=detect_disease,
            inputs=[image_input, confidence_slider],
            outputs=[annotated_image, detection_output]
        )
        
        # Info section
        gr.Markdown("""
        ---
        
        ### ℹ️ About This Model
        
        - **Framework:** YOLOv8 Object Detection
        - **Dataset:** Zindi Amini Cocoa Challenge
        - **Performance:** 
          - Precision: 75.78%
          - Recall: 66.25%
          - mAP50: 74.62%
        - **Training:** 60 epochs
        
        ### 💡 Tips
        
        1. Ensure good lighting and clear images
        2. Include the affected area in the image
        3. Adjust confidence threshold for different sensitivities
        4. Multiple detections show different disease types
        
        ### 📝 How It Works
        
        1. Upload a cocoa plant image
        2. Adjust the confidence threshold as needed
        3. Click "Detect Disease"
        4. View annotated results with bounding boxes
        5. Review detailed detection information
        
        """)
    
    return demo


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("🚀 Starting Cocoa Disease Detection Application...")
    print(f"📱 Using device: {DEVICE}")
    print("⚠️  CPU Mode: Inference may take 1-2 seconds per image")
    print("💡 Tip: For faster inference, use GPU if available")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model file '{MODEL_PATH}' not found in current directory")
        print("   Please ensure 'best.pt' is in the same directory as app.py")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=False
    )
