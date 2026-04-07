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
    
    anno_dirs = [
        project_root / 'dataset' / 'PairTally' / 'processed_dataset' / 'Anno',
        project_root / 'dataset' / 'PairTally' / 'removed' / 'Anno'
    ]

    # Map all available JSON files
    json_map = {}
    for d in anno_dirs:
        if d.exists():
            for f in d.glob('*.json'):
                json_map[f.name] = f

    print(f"Scanning {len(json_map)} annotation files...")

    updated_files_count = 0

    # Iterate through all mapped files
    for file_name, json_path in sorted(json_map.items()):
        current_data = {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_name}: {e}")
            continue

        # Only process files where loc_bbox is empty
        if current_data.get('loc_bbox') and len(current_data['loc_bbox']) > 0:
            # print(f"DEBUG: Skipping {file_name} because loc_bbox is already populated ({len(current_data['loc_bbox'])} boxes).")
            continue

        # Ignore manifest
        if file_name == 'pairtally_annotations_simple.json':
            continue

        # Determine the name of the paired JSON file
        base_name = json_path.stem
        pair_file_name = None
        if base_name.endswith('_positive'):
            pair_file_name = base_name.replace('_positive', '_negative') + '.json'
        elif base_name.endswith('_negative'):
            pair_file_name = base_name.replace('_negative', '_positive') + '.json'
        
        if not pair_file_name:
            # print(f"DEBUG: Could not determine pair for {file_name}. Skipping.")
            continue

        if pair_file_name in json_map:
            pair_json_path = json_map[pair_file_name]
            pair_data = {}
            try:
                with open(pair_json_path, 'r', encoding='utf-8') as f:
                    pair_data = json.load(f)
                if pair_data.get('loc_bbox') and len(pair_data['loc_bbox']) > 0:
                    # print(f"DEBUG: Found pair {pair_file_name} with populated loc_bbox ({len(pair_data['loc_bbox'])} boxes) for empty {file_name}. Updating...")
                    current_data['loc_bbox'] = pair_data['loc_bbox'][:] # Use slice to copy list
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=4)
                    print(f"Updated {file_name} from its pair {pair_file_name} (copied {len(pair_data['loc_bbox'])} boxes).")
                    updated_files_count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing pair file {pair_file_name} for {file_name}: {e}")
        else:
            # print(f"DEBUG: Pair {pair_file_name} for {file_name} not found in json_map.")
            pass
    
    print(f"\nFinished. Updated {updated_files_count} files with annotations copied from their pairs.")

if __name__ == "__main__":
    main()