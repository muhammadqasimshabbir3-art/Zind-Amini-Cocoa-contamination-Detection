"""
Batch processing utilities for Cocoa Disease Detection
Supports multi-GPU inference and batch submission generation
"""

import os
import torch
import pandas as pd
from typing import List, Dict, Tuple
from ultralytics import YOLO
from pathlib import Path


def process_single_image(
    img_path: str,
    model: YOLO,
    conf: float = 0.5,
    device: int = 0
) -> Dict:
    """
    Process a single image and return detection results
    
    Args:
        img_path: Path to image
        model: YOLOv8 model
        conf: Confidence threshold
        device: Device ID (CPU mode - parameter kept for compatibility)
        
    Returns:
        Dictionary with detection results
    """
    submission = {
        'Image_ID': os.path.basename(img_path),
        'class': "None",
        'confidence': None,
        'ymin': None,
        'xmin': None,
        'ymax': None,
        'xmax': None
    }

    if os.path.exists(img_path):
        try:
            detections = model.predict(
                img_path,
                conf=conf,
                verbose=False,
                device="cpu"  # Using CPU
            )

            if detections[0].boxes:
                boxes = detections[0].boxes.xyxy.cpu().numpy()
                confs = detections[0].boxes.conf.cpu().numpy()
                classes = detections[0].boxes.cls.cpu().numpy()

                # Get best detection
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes[best_idx]

                submission.update({
                    'class': detections[0].names[int(classes[best_idx])],
                    'confidence': float(confs[best_idx]),
                    'ymin': float(y1),
                    'xmin': float(x1),
                    'ymax': float(y2),
                    'xmax': float(x2)
                })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    return submission


def process_batch(args: Tuple) -> List[Dict]:
    """
    Process a batch of images (for multiprocessing)

    Args:
        args: Tuple of (img_batch, conf, gpu_id, model_path, test_dir)

    Returns:
        List of detection results
    """
    img_batch, conf, gpu_id, model_path, test_dir = args
    model = YOLO(model_path)
    results = []

    for img_id in img_batch:
        img_path = os.path.join(test_dir, img_id)
        result = process_single_image(img_path, model, conf, gpu_id)
        results.append(result)

    return results


def generate_submissions(
    test_csv: str,
    test_dir: str,
    model_path: str,
    conf_thresholds: List[float] = [0.5, 0.7, 0.9],
    output_dir: str = "./submissions"
) -> List[str]:
    """
    Generate submission files for different confidence thresholds

    Args:
        test_csv: Path to test CSV file
        test_dir: Path to test images directory
        model_path: Path to model weights
        conf_thresholds: List of confidence thresholds to try
        output_dir: Output directory for submission files

    Returns:
        List of generated submission file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    submission_files = []

    # Load model once
    model = YOLO(model_path)

    # Load test data
    test_df = pd.read_csv(test_csv)
    image_ids = test_df['Image_ID'].tolist()

    print(f"Processing {len(image_ids)} images...")

    for conf in conf_thresholds:
        print(f"\n🔍 Processing confidence threshold: {conf}")
        results = []

        for img_id in image_ids:
            result = process_single_image(
                os.path.join(test_dir, img_id),
                model,
                conf=conf
            )
            results.append(result)

        # Save submission
        submission_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, f"submission_conf_{conf}.csv")
        submission_df.to_csv(output_path, index=False)
        submission_files.append(output_path)

        print(f"✅ Saved: {output_path}")
        print(f"   - Detections found: {(submission_df['confidence'].notna()).sum()}")

    return submission_files


def process_directory(
    image_dir: str,
    model_path: str,
    conf: float = 0.5,
    output_csv: str = None
) -> List[Dict]:
    """
    Process all images in a directory

    Args:
        image_dir: Directory containing images
        model_path: Path to model weights
        conf: Confidence threshold
        output_csv: Optional output CSV file

    Returns:
        List of detection results
    """
    model = YOLO(model_path)
    results = []

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {image_dir}")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        result = process_single_image(img_path, model, conf)
        results.append(result)

    # Optionally save results
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Results saved to {output_csv}")

    return results


# Example usage
if __name__ == "__main__":
    # Example: Generate submissions with different confidence thresholds

    # Configuration
    MODEL_PATH = "best.pt"
    TEST_CSV = "Test.csv"
    TEST_DIR = "test_images/"

    # Generate submissions
    submission_files = generate_submissions(
        test_csv=TEST_CSV,
        test_dir=TEST_DIR,
        model_path=MODEL_PATH,
        conf_thresholds=[0.5, 0.7, 0.9],
        output_dir="./submissions"
    )

    print("\n" + "="*50)
    print("✅ All submissions generated!")
    print("="*50)
    for f in submission_files:
        print(f"  📄 {f}")

