import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Please install it using: pip install datasets")
    exit(1)

def main():
    """
    Syncs point annotations from CoCount-train-raw to the processed JSON files
    located in dataset/CoCount-train/processed_dataset/Anno.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Paths
    raw_data_dir = project_root / "dataset" / "CoCount-train" / "CoCount-train-raw" / "data"
    anno_dir = project_root / "dataset" / "CoCount-train" / "processed_dataset" / "Anno"

    if not raw_data_dir.exists():
        print(f"Error: Raw data directory not found at {raw_data_dir}")
        return
    if not anno_dir.exists():
        print(f"Error: Annotation directory not found at {anno_dir}")
        return

    parquet_files = [str(f.resolve()) for f in raw_data_dir.glob("*.parquet")]
    if not parquet_files:
        print(f"No parquet files found in {raw_data_dir}.")
        return

    print(f"Loading CoCount-train-raw metadata from {len(parquet_files)} files...")
    # Load dataset, removing 'image' column for performance
    try:
        ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        # Only keep necessary columns to save memory
        keep_cols = ["image_name", "pos_caption", "neg_caption", "pos_points", "neg_points"]
        ds_light = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Scanning for processed annotations and syncing points...")
    updated_count = 0

    for row in ds_light:
        original_name = row.get('image_name')
        if not original_name:
            continue
            
        stem = Path(original_name).stem
        pos_caption = row.get('pos_caption', '').strip()
        neg_caption = row.get('neg_caption', '').strip()
        pos_points = row.get('pos_points', [])
        neg_points = row.get('neg_points', [])

        # Sync positive version
        pos_path = anno_dir / f"{stem}_positive.json"
        if pos_path.exists():
            if try_update_file(pos_path, pos_caption, pos_points):
                updated_count += 1

        # Sync negative version
        neg_path = anno_dir / f"{stem}_negative.json"
        if neg_path.exists():
            if try_update_file(neg_path, neg_caption, neg_points):
                updated_count += 1

    print(f"Sync complete. Updated {updated_count} annotation files with 'points' field.")

def try_update_file(file_path, target_caption, points):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Verify caption matches to ensure correct points (handles swapped pairs in raw data)
        if data.get('class_name', '').strip() == target_caption:
            data['points'] = points
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
    return False

if __name__ == "__main__":
    main()