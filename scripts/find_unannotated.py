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
    anno_dirs = [
        project_root / 'dataset' / 'PairTally' / 'processed_dataset' / 'Anno',
        project_root / 'dataset' / 'PairTally' / 'removed' / 'Anno'
    ]

    json_files = []
    for d in anno_dirs:
        if d.exists():
            json_files.extend(list(d.glob('*.json')))

    unannotated_images = []
    total_count = 0
    completed_count = 0

    # Iterate through all JSON files in the Anno directory
    for json_path in sorted(json_files):
        if json_path.name == 'pairtally_annotations_simple.json':
            continue
            
        total_count += 1
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if loc_bbox has fewer than 10 entries
            if len(data.get('loc_bbox', [])) < 10:
                # Construct the image name from the JSON filename
                # The annotator generates filenames like 'image_name_positive.json'
                image_name = json_path.stem + ".jpg"
                unannotated_images.append(image_name)
                # if json_path.parent.name == 'Anno' and json_path.parent.parent.name == 'removed':
                #     print(f"DEBUG: Incomplete file in removed_anno_dir: {json_path.name} (loc_bbox_len: {loc_bbox_len})")
            else:
                completed_count += 1
                # if json_path.parent.name == 'Anno' and json_path.parent.parent.name == 'removed':
                #     print(f"DEBUG: Complete file in removed_anno_dir: {json_path.name} (loc_bbox_len: {loc_bbox_len})")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Could not read {json_path.name}: {e}")

    completed = total_count - len(unannotated_images)
    percent = (completed / total_count * 100) if total_count > 0 else 0

    print(f"\nOverall Progress: {completed}/{total_count} ({percent:.1f}%)")

    if not unannotated_images:
        print("Great news! All images have at least 10 annotations.")
    else:
        print(f"Found {len(unannotated_images)} incomplete files.")

if __name__ == "__main__":
    main()