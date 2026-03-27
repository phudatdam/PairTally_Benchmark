import os
import json
from pathlib import Path

def main():
    """
    Scans the processed dataset's annotation folder. For any JSON file
    with an empty 'loc_bbox', it attempts to copy annotations from its
    paired image's JSON file if available.
    """
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    anno_dir = project_root / 'dataset' / 'processed_dataset' / 'Anno'

    print(f"Scanning annotation directory: {anno_dir}")

    if not anno_dir.exists():
        print(f"Error: Annotation directory not found at {anno_dir}")
        return

    updated_files_count = 0

    # Iterate through all JSON files in the Anno directory
    for json_path in sorted(anno_dir.glob('*.json')):
        current_data = {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {json_path.name}: {e}")
            continue

        # Only process files where loc_bbox is empty
        if current_data.get('loc_bbox') and len(current_data['loc_bbox']) > 0:
            continue

        # Determine the name of the paired JSON file
        base_name = json_path.stem
        pair_file_name = None
        if base_name.endswith('_positive'):
            pair_file_name = base_name.replace('_positive', '_negative') + '.json'
        elif base_name.endswith('_negative'):
            pair_file_name = base_name.replace('_negative', '_positive') + '.json'
        
        if not pair_file_name:
            # This case should ideally not happen if create_dataset.py is used
            print(f"Warning: Could not determine pair for {json_path.name}. Skipping.")
            continue

        pair_json_path = anno_dir / pair_file_name
        
        if pair_json_path.exists():
            pair_data = {}
            try:
                with open(pair_json_path, 'r', encoding='utf-8') as f:
                    pair_data = json.load(f)
                if pair_data.get('loc_bbox') and len(pair_data['loc_bbox']) > 0:
                    current_data['loc_bbox'] = pair_data['loc_bbox']
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=4)
                    print(f"Updated {json_path.name} from its pair {pair_json_path.name}")
                    updated_files_count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing pair file {pair_json_path.name} for {json_path.name}: {e}")
    
    print(f"\nFinished. Updated {updated_files_count} files with annotations copied from their pairs.")

if __name__ == "__main__":
    main()