import json
import os
import shutil
from pathlib import Path

def main():
    # Define paths relative to the project root
    # Assuming the script is run from the project root directory
    base_dataset_dir = Path('dataset/pairtally_dataset')
    images_dir = base_dataset_dir / 'images'
    annotations_file = base_dataset_dir / 'annotations' / 'pairtally_annotations_simple.json'
    
    # Output directories
    output_image_dir = Path('dataset/processed_dataset/Image')
    output_anno_dir = Path('dataset/processed_dataset/Anno')
    
    # Create output directories if they don't exist
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_anno_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if annotations file exists
    if not annotations_file.exists():
        print(f"Error: Annotation file not found at {annotations_file}")
        return

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    print(f"Found {len(annotations)} entries in annotations file.")

    processed_count = 0
    
    for img_name, data in annotations.items():
        src_img_path = images_dir / img_name
        
        # Check if source image exists
        if not src_img_path.exists():
            print(f"Warning: Image {img_name} not found in {images_dir}. Skipping.")
            continue
            
        # Get class names from metadata
        positive_class = data.get('positive_prompt', 'positive')
        negative_class = data.get('negative_prompt', 'negative')
        
        # Define the two variations
        variations = [
            ('positive', positive_class),
            ('negative', negative_class)
        ]
        
        for suffix, class_name in variations:
            # Construct new filename
            # Preserving original extension (likely .jpg)
            new_img_name = f"{src_img_path.stem}_{suffix}{src_img_path.suffix}"
            new_json_name = f"{src_img_path.stem}_{suffix}.json"
            
            dest_img_path = output_image_dir / new_img_name
            dest_json_path = output_anno_dir / new_json_name
            
            # Safety check: Don't overwrite if the JSON already has annotations
            if dest_json_path.exists():
                try:
                    with open(dest_json_path, 'r') as f:
                        existing = json.load(f)
                    if existing.get('loc_bbox') and len(existing['loc_bbox']) > 0:
                        # Skip this file to preserve manual work
                        continue
                except:
                    pass

            # Copy image
            shutil.copy2(src_img_path, dest_img_path)
            
            # Create JSON annotation content
            # loc_bbox is initialized as empty list, to be populated by annotation tool later
            anno_content = {
                "class_name": class_name,
                "loc_bbox": [], 
                "exam_bbox": [],
                "source_img_name": img_name
            }
            
            # Write JSON file
            with open(dest_json_path, 'w') as json_file:
                json.dump(anno_content, json_file, indent=4)
                
        processed_count += 1
        
    print(f"Successfully processed {processed_count} original images.")
    print(f"Generated {processed_count * 2} images in '{output_image_dir}' and {processed_count * 2} JSON files in '{output_anno_dir}'.")

if __name__ == '__main__':
    main()
