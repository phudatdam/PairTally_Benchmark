import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset

# Point to the folder where you downloaded the CLI files
local_dir = Path(__file__).parent.parent / "dataset" / "CoCount-train"
local_data_dir = local_dir / "data"

if not local_dir.exists():
    raise FileNotFoundError(f"The directory {local_dir} does not exist. Please run the CLI download command first.")

# Use rglob to find all parquet files recursively. 
parquet_files = [str(f.resolve()) for f in local_data_dir.glob("*.parquet")]

if not parquet_files:
    raise FileNotFoundError(f"No .parquet files found in {local_data_dir}. Check if the CLI download finished successfully.")

print(f"Found {len(parquet_files)} parquet files. Loading...")
ds = load_dataset("parquet", data_files={"train": parquet_files})

print("Dataset loaded successfully!")
print(ds)

# --- Inspection Logic ---

manifest_path = Path(__file__).parent.parent / 'dataset' / 'pairtally_dataset' / 'annotations' / 'pairtally_annotations_simple.json'

if manifest_path.exists():
    print(f"\nComparing with local manifest: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        local_manifest = json.load(f)
    
    local_image_names = set(local_manifest.keys())
    hf_train = ds['train']
    
    # Create a mapping of image_name to index for fast lookup
    print("Building index mapping for CoCount dataset...")
    hf_image_names = hf_train['image_name']
    
    # Normalization logic to handle the different ordering of segments
    def normalize_name(name):
        name = name.lower()
        stem = Path(name).stem
        
        # Split and clean
        segments = stem.split('_')
        normalized = []
        
        for s in segments:
            if s == 'inter': 
                continue
            # Remove integer padding: '00068' -> '68'
            if s.isdigit():
                normalized.append(str(int(s)))
            # Ignore likely 6-char hex hashes (e.g., 'f29cfe')
            elif len(s) == 6 and all(c in '0123456789abcdef' for c in s):
                continue
            else:
                normalized.append(s)
        
        return set(normalized)

    print("Normalizing names for robust matching...")
    hf_map = {tuple(sorted(normalize_name(n))): n for n in hf_image_names}
    
    # For the local manifest, we search for a matching set of segments
    matches = []
    for local_name in local_image_names:
        local_norm = tuple(sorted(normalize_name(local_name)))
        
        # Use a threshold-based segment match
        local_set = set(local_norm)
        for hf_norm, hf_orig in hf_map.items():
            hf_set = set(hf_norm)
            # If category, video_id, and at least one frame number match
            intersection = local_set.intersection(hf_set)
            # High confidence if most segments (excluding the scenario) match
            if len(intersection) >= min(len(local_set), len(hf_set)) - 1:
                matches.append((local_name, hf_orig))
                break
    
    print(f"Total images in local manifest: {len(local_image_names)}")
    print(f"Matches found via segment analysis: {len(matches)}")
    
    if matches:
        local_sample, hf_sample = matches[0]
        print(f"\nMatch found:")
        print(f"  Local: {local_sample}")
        print(f"  HF:    {hf_sample}")
        
        # Find the row in HF dataset
        idx = hf_image_names.index(hf_sample)
        hf_row = hf_train[idx]
        local_row = local_manifest[local_sample]
        
        print(f"HF Category: {hf_row['category']}")
        print(f"HF Pos Count: {hf_row['pos_count']} | Local Points Count: {len(local_row.get('points', []))}")
        print(f"HF Neg Count: {hf_row['neg_count']} | Local Neg Points Count: {len(local_row.get('negative_points', []))}")
        print(f"HF Positive Caption: {hf_row['pos_caption']}")
        
        # Check if points match (optional sanity check)
        hf_pos_points = hf_row['pos_points']
        local_pos_points = local_row.get('points', [])
        print(f"Match quality: {len(hf_pos_points) == len(local_pos_points)} (Point counts match)")
else:
    print(f"\nManifest not found at {manifest_path}. Skipping cross-reference.")