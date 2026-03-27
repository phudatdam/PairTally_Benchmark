import os
import json
from pathlib import Path

def main():
    """
    Scans the processed dataset's annotation folder and prints the names of images
    that have fewer than 10 'loc_bbox' entries.
    """
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    anno_dir = project_root / 'dataset' / 'processed_dataset' / 'Anno'

    if not anno_dir.exists():
        print(f"Error: Annotation directory not found at {anno_dir}")
        return

    unannotated_images = []

    # Iterate through all JSON files in the Anno directory
    for json_path in sorted(anno_dir.glob('*.json')):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if loc_bbox has fewer than 10 entries
            if len(data.get('loc_bbox', [])) < 10:
                # Construct the image name from the JSON filename
                # The annotator generates filenames like 'image_name_positive.json'
                image_name = json_path.stem + ".jpg"
                unannotated_images.append(image_name)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Could not read {json_path.name}: {e}")

    if not unannotated_images:
        print("Great news! All images in the active folder have at least 10 annotations.")
    else:
        print(f"Found {len(unannotated_images)} images with fewer than 10 annotations:")
        for img in unannotated_images:
            print(img)

if __name__ == "__main__":
    main()