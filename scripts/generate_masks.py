import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
import cv2
import numpy as np
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: 'segment-anything' library not found.")
    print("Please install it using: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

def main():
    """
    Reads images and their corresponding 'exam_bbox' from JSON annotations,
    runs SAM to generate segmentation masks, and saves them to a new folder.
    """
    # 1. Setup Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    image_dir = project_root / 'dataset' / 'processed_dataset' / 'Image'
    anno_dir = project_root / 'dataset' / 'processed_dataset' / 'Anno'
    mask_root = project_root / 'dataset' / 'processed_dataset' / 'mask'

    # 2. SAM Configuration
    # Ensure you have downloaded the checkpoint file (e.g., vit_h)
    # Link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    checkpoint_path = project_root / 'models' / 'sam_vit_h_4b8939.pth'
    model_type = "vit_h"

    if not checkpoint_path.exists():
        print(f"Error: SAM checkpoint not found at {checkpoint_path}")
        print(f"Please create the directory '{project_root / 'models'}' and place the weights there.")
        return

    # 3. Initialize Model and Predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SAM model ({model_type})...")
    try:
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Failed to initialize SAM: {e}")
        return

    # 4. Prepare Output Directory
    mask_root.mkdir(parents=True, exist_ok=True)

    # 5. Process Images
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images. Starting segmentation...")

    for img_path in image_files:
        # Find corresponding JSON
        json_path = anno_dir / (img_path.stem + ".json")
        
        if not json_path.exists():
            print(f"Warning: Annotation file missing for {img_path.name}. Skipping.")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        exam_bboxes = data.get('exam_bbox', [])
        if not exam_bboxes:
            print(f"No 'exam_bbox' found for {img_path.name}. Skipping.")
            continue

        # Prepare subfolder for masks
        img_mask_dir = mask_root / img_path.stem
        img_mask_dir.mkdir(parents=True, exist_ok=True)

        # Load Image using OpenCV (BGR) then convert to RGB for SAM
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Error: Could not read image {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Set the image in the predictor once
        predictor.set_image(image_rgb)

        # Generate masks for each bounding box
        for i, bbox in enumerate(exam_bboxes):
            # Predict mask using the box prompt [xmin, ymin, xmax, ymax]
            input_box = np.array(bbox)
            masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

            # Convert boolean mask to 0-255 uint8 image
            mask_uint8 = (masks[0] * 255).astype(np.uint8)
            cv2.imwrite(str(img_mask_dir / f"mask{i}.png"), mask_uint8)

        print(f"Generated {len(exam_bboxes)} masks for {img_path.name}")

    print(f"\nProcessing complete. Masks are saved in: {mask_root}")

if __name__ == "__main__":
    main()