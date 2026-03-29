import json
from pathlib import Path
import os

def count_files_in_directory(directory_path, extensions=None):
    """Counts files in a directory, optionally filtering by extension."""
    if not directory_path.is_dir():
        return 0, f"Directory not found: {directory_path}"
    
    count = 0
    for item in directory_path.iterdir():
        if item.is_file():
            if extensions:
                if item.suffix.lower() in extensions:
                    count += 1
            else:
                count += 1
    return count, None

def main(trash_path=None):
    print("--- Dataset File Count and Integrity Check ---")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    trash_dir = Path(trash_path) if trash_path else None
    
    # Path to the main annotation manifest
    json_manifest_path = project_root / 'dataset' / 'pairtally_dataset' / 'annotations' / 'pairtally_annotations_simple.json'
    
    # Expected total image count from the manifest (as per your information)
    expected_total_images_in_manifest = 681

    # Paths to relevant directories based on your project structure
    source_image_dir = project_root / 'dataset' / 'pairtally_dataset' / 'images'
    processed_image_dir = project_root / 'dataset' / 'processed_dataset' / 'Image'
    processed_anno_dir = project_root / 'dataset' / 'processed_dataset' / 'Anno'
    removed_image_dir = project_root / 'dataset' / 'removed' / 'Image'
    removed_anno_dir = project_root / 'dataset' / 'removed' / 'Anno'
    
    manifest_image_names = [] # Renamed for clarity
    # 1. Count entries in the main JSON manifest
    print(f"\nChecking main annotation manifest: {json_manifest_path}")
    if not json_manifest_path.exists():
        print(f"Error: JSON manifest not found at {json_manifest_path}")
        total_images_in_manifest = 0
    else:
        try:
            with open(json_manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            manifest_image_names = list(manifest_data.keys()) # Store full image names
            total_images_in_manifest = len(manifest_image_names) # Use the list length
            print(f"Found {total_images_in_manifest} image entries in the main manifest.")

            if total_images_in_manifest == expected_total_images_in_manifest:
                print(f"Manifest count matches expected total of {expected_total_images_in_manifest} images.")
            elif total_images_in_manifest < expected_total_images_in_manifest:
                missing_in_manifest = expected_total_images_in_manifest - total_images_in_manifest
                print(f"Warning: {missing_in_manifest} image entries are missing from the manifest compared to the expected {expected_total_images_in_manifest}.")
                print("This suggests that the corresponding annotation files might have been lost or the dataset is incomplete.")
            else:
                extra_in_manifest = total_images_in_manifest - expected_total_images_in_manifest
                print(f"Warning: {extra_in_manifest} extra image entries found in the manifest compared to the expected {expected_total_images_in_manifest}.")
                print("This might indicate an issue with the expected count or duplicate entries.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_manifest_path.name}: {e}")
            total_images_in_manifest = 0
        except IOError as e:
            print(f"Error reading {json_manifest_path.name}: {e}")
            total_images_in_manifest = 0

    # 2. Count files in various directories
    print("\nCounting files in dataset folders:")

    folders_to_check = [
        (source_image_dir, ['.jpg', '.jpeg', '.png'], "Source Images (Original)"),
        (processed_image_dir, ['.jpg', '.jpeg', '.png'], "Processed Images"),
        (processed_anno_dir, ['.json'], "Processed Annotations"),
        (removed_image_dir, ['.jpg', '.jpeg', '.png'], "Removed Images"),
        (removed_anno_dir, ['.json'], "Removed Annotations")
    ]

    for path, extensions, description in folders_to_check:
        count, error = count_files_in_directory(path, extensions)
        if error:
            print(f"- {description}: {error}")
        else:
            print(f"- {description}: {count} files found.")

    # 3. Identify exactly what is missing and what is in trash
    if manifest_image_names:
        print("\n--- Missing File Audit ---")
        
        # Build sets of what we have in the actual dataset folders
        source_images = set()
        if source_image_dir.exists():
            source_images.update([f.name for f in source_image_dir.iterdir() if f.is_file()])

        existing_images = set()
        for d in [processed_image_dir, removed_image_dir]:
            if d.exists():
                existing_images.update([f.name for f in d.iterdir() if f.is_file()])
        
        existing_annos = set()
        for d in [processed_anno_dir, removed_anno_dir]:
            if d.exists():
                existing_annos.update([f.name for f in d.iterdir() if f.suffix == '.json'])

        # Build sets of what is in the trash
        trash_images = set()
        trash_annos = set()
        trash_removed_txt_found = False
        trash_weird_txt_found = False

        if trash_dir:
            print(f"Checking trash directory: {trash_dir}")
            if not trash_dir.exists():
                print(f"WARNING: Trash directory '{trash_dir}' does not exist.")
            else:
                found_in_trash_count = 0
                for f in trash_dir.rglob('*'):
                    if f.is_file():
                        found_in_trash_count += 1
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            trash_images.add(f.name)
                        elif f.suffix.lower() == '.json':
                            trash_annos.add(f.name)
                        elif f.name == 'removed.txt':
                            trash_removed_txt_found = True
                        elif f.name == 'weird_bbox.txt':
                            trash_weird_txt_found = True
                print(f"Found {found_in_trash_count} files recursively in trash directory.")
                print(f"  - {len(trash_images)} image files (by name)")
                print(f"  - {len(trash_annos)} annotation files (by name)")

        missing_source_imgs = [k for k in manifest_image_names if k not in source_images]
        
        missing_processed_imgs = []
        recoverable_imgs = []
        for img_name in manifest_image_names:
            stem = Path(img_name).stem
            suffix = Path(img_name).suffix
            for variant in [f"{stem}_positive{suffix}", f"{stem}_negative{suffix}"]:
                if variant in existing_images:
                    continue
                if variant in trash_images:
                    recoverable_imgs.append(variant)
                else:
                    missing_processed_imgs.append(variant)
        
        missing_annos = []
        recoverable_annos = []
        for img_name in manifest_image_names:
            base_stem = Path(img_name).stem
            positive_json_name = f"{base_stem}_positive.json"
            negative_json_name = f"{base_stem}_negative.json"
            
            # Check positive JSON
            if positive_json_name not in existing_annos:
                if positive_json_name in trash_annos:
                    recoverable_annos.append(positive_json_name)
                else:
                    missing_annos.append(positive_json_name)
            
            # Check negative JSON
            if negative_json_name not in existing_annos:
                if negative_json_name in trash_annos:
                    recoverable_annos.append(negative_json_name)
                else:
                    missing_annos.append(negative_json_name)

        print(f"Source Images: {len(source_images)} / {expected_total_images_in_manifest} present.")
        print(f"Processed:     {len(existing_images)} active, {len(recoverable_imgs)} in trash, {len(missing_processed_imgs)} GONE.")
        print(f"Annotations: {len(existing_annos)} active, {len(recoverable_annos)} in trash, {len(missing_annos)} GONE.")

        if trash_dir and recoverable_annos:
            print(f"\nSUCCESS: Found {len(recoverable_annos)} annotation files in trash!")
            print("You can restore these to 'dataset/processed_dataset/Anno/'")

        if missing_annos:
            print(f"\nWARNING: {len(missing_annos)} annotations are still missing.")
            if len(missing_annos) < 15:
                for m in missing_annos: print(f"  - {m}")

        # Add a summary for total expected vs found
        total_expected_images = len(manifest_image_names)
        total_expected_processed = total_expected_images * 2 # Each image has a positive and negative JSON

        total_found_processed_images = len(existing_images) + len(recoverable_imgs)
        total_found_annos = len(existing_annos) + len(recoverable_annos)

        print(f"\nSummary:")
        print(f"  Expected Processed Images: {total_expected_processed}, Found: {total_found_processed_images} (Missing: {total_expected_processed - total_found_processed_images})")
        print(f"  Expected Annotations:      {total_expected_processed}, Found: {total_found_annos} (Missing: {total_expected_processed - total_found_annos})")

        print("\n--- Metadata File Audit ---")
        if (project_root / 'dataset' / 'removed.txt').exists():
            print(f"SUCCESS: 'removed.txt' found in '{project_root / 'dataset'}'")
        else:
            print(f"WARNING: 'removed.txt' NOT found in trash. This file tracks removed images.")
        
        if (project_root / 'dataset' / 'weird_bbox.txt').exists():
            print(f"SUCCESS: 'weird_bbox.txt' found in '{project_root / 'dataset'}'")
        else:
            print(f"WARNING: 'weird_bbox.txt' NOT found in trash. This file tracks images with weird bboxes.")

    print("\n--- End of Check ---")
    if not trash_path:
        print("\nTIP: Run this script with the path to your downloaded trash to see what is recoverable:")
        print("python scripts/check_dataset_counts.py \"C:/Path/To/Downloaded/Trash\"")

if __name__ == "__main__":
    import sys
    t_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(t_path)