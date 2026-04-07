import os
import json
from pathlib import Path
from datasets import load_dataset, Image as HFImage

# Point to the folder where you downloaded the CoCount-test files
local_dir = Path(__file__).parent.parent / "dataset" / "CoCount-test" / "CoCount-test-raw"
local_data_dir = local_dir / "data"

if not local_dir.exists():
    print(f"Error: The directory {local_dir} does not exist.")
    print("Please run the following command first:")
    print("python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='BBVisual/CoCount-test', repo_type='dataset', local_dir='dataset/CoCount-test/CoCount-test-raw', local_dir_use_symlinks=False)\"")
    exit(1)

# Use rglob to find all parquet files recursively. 
parquet_files = [str(f.resolve()) for f in local_data_dir.glob("*.parquet")]

if not parquet_files:
    raise FileNotFoundError(f"No .parquet files found in {local_data_dir}. Check if the download finished successfully.")

print(f"Found {len(parquet_files)} parquet files in CoCount-test. Loading...")
ds = load_dataset("parquet", data_files={"test": parquet_files})

print("Test dataset loaded successfully!")
print(ds)

# --- Inspection Logic ---

manifest_path = Path(__file__).parent.parent / 'dataset' / 'PairTally' / 'pairtally_dataset' / 'annotations' / 'pairtally_annotations_simple.json'

if manifest_path.exists():
    print(f"\nComparing with local manifest: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        local_manifest = json.load(f)
    
    local_image_names = set(local_manifest.keys())
    # Identify the split key (usually 'test' or 'train' depending on parquet metadata)
    split_key = list(ds.keys())[0]
    hf_data = ds[split_key]

    # To access filenames without loading pixel data, we cast the column to disable decoding.
    hf_data_raw = hf_data.cast_column("image", HFImage(decode=False))
    hf_image_names = [img_info['path'] if img_info['path'] else "" for img_info in hf_data_raw['image']]
    
    def normalize_name(name):
        if not name: return set()
        name = name.lower()
        stem = Path(name).stem
        segments = stem.split('_')
        normalized = []
        for s in segments:
            if s in ['inter', 'intra']: continue
            if s.isdigit(): normalized.append(str(int(s)))
            elif len(s) == 6 and all(c in '0123456789abcdef' for c in s): continue
            else: normalized.append(s)
        return set(normalized)

    def get_hf_mapping(dataset, names_list):
        mapping = {}
        vids = dataset['video_id'] if 'video_id' in dataset.column_names else [None]*len(names_list)
        cats = dataset['category'] if 'category' in dataset.column_names else [None]*len(names_list)
        pos_c = dataset['pos_count'] if 'pos_count' in dataset.column_names else [0]*len(names_list)
        neg_c = dataset['neg_count'] if 'neg_count' in dataset.column_names else [0]*len(names_list)
        points = dataset['pos_points'] if 'pos_points' in dataset.column_names else [[]]*len(names_list)
        
        for i, (name, vid, cat, p, n, pts) in enumerate(zip(names_list, vids, cats, pos_c, neg_c, points)):
            norm_set = normalize_name(name)
            if vid: norm_set.update(normalize_name(vid))
            if cat: norm_set.update(normalize_name(cat))
            key = tuple(sorted(norm_set))
            if key not in mapping: mapping[key] = []
            mapping[key].append({'name': name, 'idx': i, 'pos': p, 'neg': n, 'has_points': len(pts) > 0})
        return mapping

    def find_all_matches(local_names, all_mappings, manifest_dict):
        final_results = {} # local_name -> (split, hf_info)
        for local_name in local_names:
            local_set = normalize_name(local_name)
            local_p = len(manifest_dict[local_name].get('points', []))
            local_n = len(manifest_dict[local_name].get('negative_points', []))
            
            best_match = None
            for split, mapping in all_mappings.items():
                for hf_norm, candidates in mapping.items():
                    if len(local_set.intersection(set(hf_norm))) >= min(len(local_set), len(hf_norm)) - 1:
                        for cand in candidates:
                            # Exact count match is preferred
                            if cand['pos'] == local_p and cand['neg'] == local_n:
                                # Priority: Split with points > split without points
                                if not best_match or (cand['has_points'] and not best_match[1]['has_points']):
                                    best_match = (split, cand)
                            # If the split is "train", it's high priority because of points
                            if split == "train" and cand['has_points']:
                                best_match = (split, cand)
            if best_match:
                final_results[local_name] = best_match
        return final_results

    # Build global map across all splits
    all_splits_map = {"test": get_hf_mapping(hf_data, hf_image_names)}
    for sn in ["val", "train"]:
        sp = Path(__file__).parent.parent / "dataset" / f"CoCount-{sn}" / f"CoCount-{sn}-raw" / "data"
        if sp.exists():
            s_files = [str(f.resolve()) for f in sp.glob("*.parquet")]
            s_ds = load_dataset("parquet", data_files={sn: s_files}, split=sn)
            # Unify naming extraction
            s_names = s_ds['image_name'] if 'image_name' in s_ds.column_names else [Path(i['path']).name for i in s_ds.cast_column("image", HFImage(decode=False))['image']]
            all_splits_map[sn] = get_hf_mapping(s_ds, s_names)

    matches = find_all_matches(local_image_names, all_splits_map, local_manifest)
    print(f"Global Match Result: {len(matches)} / {len(local_image_names)} matched.")
    
    train_matches = [name for name, (split, info) in matches.items() if split == "train"]
    print(f"Images with available point annotations: {len(train_matches)}")

    # Identify the specific 4 images found in train that were missing from test
    test_mapping = all_splits_map.get("test", {})
    new_from_train = []
    for name in train_matches:
        local_set = normalize_name(name)
        found_in_test = False
        for hf_norm in test_mapping.keys():
            if len(local_set.intersection(set(hf_norm))) >= min(len(local_set), len(hf_norm)) - 1:
                found_in_test = True
                break
        if not found_in_test:
            new_from_train.append(name)

    if new_from_train:
        print(f"\nNames of the {len(new_from_train)} images derived ONLY from CoCount-train:")
        for name in sorted(new_from_train):
            print(f"  - {name}")

    # --- Swapped Pair Inspection ---
    if hf_data is not None and 'video_id' in hf_data.column_names:
        # Count unique video IDs across test to check overlap
        test_vids = set(hf_data['video_id'])
        
        print("\nChecking for Swapped Pairs in Test Set...")
        v_stats = {}
        for v, p, n in zip(hf_data['video_id'], hf_data['pos_count'], hf_data['neg_count']):
            if v not in v_stats: v_stats[v] = []
            v_stats[v].append((p, n))
        sw_count = sum(1 for v, c in v_stats.items() if len(c) > 1 and any(c[0][0] == x[1] and c[0][1] == x[0] for x in c[1:]))
        print(f"Found {sw_count} instances where counts are swapped for the same video_id.")

    if matches:
        # Get a predictable sample from the results dictionary
        local_sample = sorted(matches.keys())[0]
        split_name, info = matches[local_sample]
        
        # Load the specific row from the correct split for display
        if split_name == "test":
            target_ds = hf_data
        else:
            sp = Path(__file__).parent.parent / "dataset" / f"CoCount-{split_name}" / f"CoCount-{split_name}-raw" / "data"
            s_files = [str(f.resolve()) for f in sp.glob("*.parquet")]
            target_ds = load_dataset("parquet", data_files={split_name: s_files}, split=split_name)
            
        hf_row = target_ds[info['idx']]
        local_row = local_manifest[local_sample]
        
        print(f"\nExample Match (from {split_name} split): {local_sample} -> {info['name']}")
        print(f"HF Category: {hf_row['category']}")
        print(f"HF Pos Count: {hf_row['pos_count']} | Local Points Count: {len(local_row.get('points', []))}")
        print(f"HF Positive Caption: {hf_row.get('pos_caption', 'N/A')}")
        
        if not hf_row['pos_points']:
            print(f"Note: 'pos_points' is EMPTY in the {split_name} split.")
        else:
            print(f"SUCCESS: Points ARE available in the {split_name} split!")

    missing_names = local_image_names - set(matches.keys())
    if not missing_names:
        print("\nAll 681 images accounted for across Test, Val, and Train splits.")
    else:
        missing_log = Path(__file__).parent.parent / "dataset" / "PairTally" / "missing_images.txt"
        with open(missing_log, 'w') as f:
            f.write('\n'.join(sorted(list(missing_names))))
        print(f"\nStill missing {len(missing_names)} images. Names saved to {missing_log}")
else:
    print(f"\nManifest not found at {manifest_path}. Skipping cross-reference.")