import os
import json
from pathlib import Path
from datasets import load_dataset
from collections import Counter, defaultdict

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    local_dir = project_root / "dataset" / "CoCount-train"
    local_data_dir = local_dir / "data"
    
    # 0. Load Metadata for strict mapping if available
    metadata_path = project_root / "dataset" / "pairtally_dataset" / "annotations" / "image_metadata.json"
    meta_map = {}
    if metadata_path.exists():
        print(f"Loading metadata mapping from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f) # meta_data is a dict of image_name -> info
            
            processed_ids = set() # To track (super_cat_id, cat_id, attr_id) tuples
            for img_key, img_info in meta_data.items():
                super_c_code = img_info.get('super_category_id', '').strip().upper()
                c_code = img_info.get('category_id', '').strip().upper()
                c_name = img_info.get('category', 'Unknown').strip().lower()
                attr_name = img_info.get('attribute', c_name).strip().lower()
                attr_id = img_info.get('category_attribute_id', 'UNK').strip().upper()

                # Add simple category mapping (e.g., BEA0 -> bean) for fallback
                if c_code and (None, c_code, '0') not in processed_ids:
                    meta_map[f"{c_code}0"] = c_name
                    processed_ids.add((None, c_code, '0'))
                # Add combined category mapping (e.g., FOO_BEA0 -> bean)
                if super_c_code and c_code and (super_c_code, c_code, '0') not in processed_ids:
                    meta_map[f"{super_c_code}_{c_code}0"] = c_name
                    processed_ids.add((super_c_code, c_code, '0'))
                
                # Add simple subcategory mapping (e.g., BEA1 -> black bean) for fallback
                if attr_id and (None, attr_id, '') not in processed_ids:
                    meta_map[attr_id] = attr_name
                    processed_ids.add((None, attr_id, ''))
                # Add combined subcategory mapping (e.g., FOO_BEA1 -> black bean)
                if super_c_code and attr_id and (super_c_code, attr_id, '') not in processed_ids:
                    meta_map[f"{super_c_code}_{attr_id}"] = attr_name
                    processed_ids.add((super_c_code, attr_id, ''))

    if not local_data_dir.exists():
        print(f"Error: The directory {local_data_dir} does not exist.")
        return

    parquet_files = [str(f.resolve()) for f in local_data_dir.glob("*.parquet")]
    if not parquet_files:
        print(f"No .parquet files found in {local_data_dir}.")
        return

    print(f"Loading {len(parquet_files)} parquet files from CoCount-train...")
    ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
    
    # We remove the 'image' column to save memory and processing time
    ds_light = ds.remove_columns(["image"])

    total_images = 0
    total_objects = 0
    
    # High-level distributions
    type_counts = Counter()        # INTER vs INTRA
    supercat_counts = Counter()    # Food, Fun, etc.
    
    # Global stats based on IDs to ensure uniqueness (e.g. Prefix 'PAS')
    cat_img_counts = Counter()
    cat_obj_counts = Counter()
    
    # Global stats based on IDs to ensure uniqueness (e.g. Code 'PAS1')
    subcat_img_counts = Counter()
    subcat_obj_counts = Counter()

    code_to_name = {}
    
    supercat_map = {
        "FOO": "Food",
        "FUN": "Fun",
        "HOU": "Household",
        "OFF": "Office",
        "OTR": "Other"
    }

    print("Iterating over dataset to aggregate statistics...")

    for row in ds_light:
        total_images += 1
        
        # 1. Extract Structural Info
        # The 'category' field in the parquet often contains 'TYPE_SUPERCAT_CAT1_CAT2' 
        # e.g., 'INTER_FOO_BEA1_SED2'. We split this to get the real metadata.
        meta_str = row.get("category", "")
        parts = meta_str.split('_') # [INTER, FOO, BEA1, SED2]
        
        img_type = parts[0] if len(parts) > 0 else "Unknown"
        type_counts[img_type] += 1
        
        super_code = parts[1] if len(parts) > 1 else "Unknown"
        super_name = supercat_map.get(super_code, f"Unknown ({super_code})")
        supercat_counts[super_name] += 1
        
        # 2. Extract Object Names and Counts
        pos_sub = row.get("pos_caption", "Unknown").strip()
        neg_sub = row.get("neg_caption", "Unknown").strip()
        
        pos_n = row.get("pos_count", 0)
        neg_n = row.get("neg_count", 0)
        total_objects += (pos_n + neg_n)
        
        # 3. Identify Codes
        pos_code = parts[2] if len(parts) > 2 else "UNK?"
        neg_code = parts[3] if len(parts) > 3 else "UNK?"
        
        # Construct unique keys by combining super_code with category/subcategory codes.
        # Fallback to original code if super_code is "Unknown" (less robust but prevents errors).
        # Category keys end with '0' (e.g., FOO_BEA0), subcategory keys are full codes (e.g., FOO_BEA1).
        cat_key_pos = f"{super_code}_{pos_code[:3]}0" if super_code != "Unknown" else f"{pos_code[:3]}0"
        cat_key_neg = f"{super_code}_{neg_code[:3]}0" if super_code != "Unknown" else f"{neg_code[:3]}0"

        subcat_key_pos = f"{super_code}_{pos_code}" if super_code != "Unknown" else pos_code
        subcat_key_neg = f"{super_code}_{neg_code}" if super_code != "Unknown" else neg_code

        # Update Mappings (Metadata file takes precedence over row captions)
        if subcat_key_pos not in code_to_name:
            code_to_name[subcat_key_pos] = meta_map.get(subcat_key_pos, pos_sub)
        if subcat_key_neg not in code_to_name:
            code_to_name[subcat_key_neg] = meta_map.get(subcat_key_neg, neg_sub)

        # 4. Aggregation Logic (Using IDs as keys to prevent duplication)
        # Category stats (Grouped by combined prefix + '0')
        cat_img_counts[cat_key_pos] += 1
        cat_obj_counts[cat_key_pos] += pos_n
        if cat_key_pos != cat_key_neg:
            cat_img_counts[cat_key_neg] += 1
        cat_obj_counts[cat_key_neg] += neg_n

        # Subcategory stats (Grouped by combined full code)
        subcat_img_counts[subcat_key_pos] += 1
        subcat_obj_counts[subcat_key_pos] += pos_n
        if subcat_key_pos != subcat_key_neg:
            subcat_img_counts[subcat_key_neg] += 1
        subcat_obj_counts[subcat_key_neg] += neg_n

    print("\n" + "="*40)
    print(" COCOUNT-TRAIN DATASET ANALYSIS")
    print("="*40)
    
    # 4. Filter and organize the mapping
    # Group codes by their 3-letter prefix for a cleaner mapping display
    grouped_mappings = defaultdict(dict)
    for full_key, name in code_to_name.items():
        parts = full_key.split('_')
        if len(parts) == 2:
            super_prefix = parts[0]
            sub_code_part = parts[1]
            prefix = sub_code_part[:3]
            combined_prefix_key = f"{super_prefix}_{prefix}"
            grouped_mappings[combined_prefix_key][full_key] = name
        else:
            prefix = full_key[:3]
            grouped_mappings[prefix][full_key] = name

    # 5. Calculate specialized counts and adjust for paired samples
    # Since each physical image has 2 samples (A+B and B+A), we divide by 2
    # to show the count of unique visual scenes/object instances.
    adj_total_images = total_images // 2
    adj_total_objects = total_objects // 2

    unique_prefixes = list(cat_img_counts.keys())
    all_codes = list(subcat_img_counts.keys())
    general_codes = [c for c in all_codes if c.endswith('0')]
    actual_sub_codes = [c for c in all_codes if not c.endswith('0')]

    print(f"\n[BASIC STATS]")
    print(f"Total Unique Images (Pairs): {adj_total_images}")
    print(f"Total Unique Objects:        {adj_total_objects}")
    print(f"Avg Objects/Image:           {adj_total_objects/adj_total_images:.2f}" if adj_total_images > 0 else "N/A")
    print(f"Total Unique Category Groups (Prefixes): {len(unique_prefixes)}")
    print(f"Total Unique Codes Encountered:          {len(all_codes)}")
    print(f"  - General Category Labels (suffix 0):  {len(general_codes)}")
    print(f"  - Actual Sub-variant Labels (suffix 1/2): {len(actual_sub_codes)}")
    print(f"(Note: Stats above adjusted for paired samples)")

    def get_display_name(key):
        # Try exact key (FOO_BEA0)
        if key in meta_map: return meta_map[key]
        if key in code_to_name: return code_to_name[key]
        # Try fallback to simple code (BEA0)
        code_only = key.split('_')[-1] if '_' in key else key
        if code_only in meta_map: return meta_map[code_only]
        if code_only in code_to_name: return code_to_name[code_only]
        return key

    print(f"\n[CODE TO NAME MAPPING]")
    for combined_key in sorted(grouped_mappings.keys()):
        super_prefix = combined_key.split('_')[0] if '_' in combined_key else ""
        prefix = combined_key.split('_')[-1] if '_' in combined_key else combined_key
        
        # Print Cat(0) then Subcats(1,2)
        cat = get_display_name(f"{super_prefix}_{prefix}0" if super_prefix else f"{prefix}0").capitalize()
        sub1 = get_display_name(f"{super_prefix}_{prefix}1" if super_prefix else f"{prefix}1").capitalize()
        sub2 = get_display_name(f"{super_prefix}_{prefix}2" if super_prefix else f"{prefix}2").capitalize()
        
        print(f"  {combined_key:8} | Cat: {cat:30} | Sub1: {sub1:30} | Sub2: {sub2}")

    print(f"\n[TYPE DISTRIBUTION (INTER/INTRA)]")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t:10}: {count // 2:6} images")

    print(f"\n[SUPERCATEGORY (FOO/FUN/HOU/OFF/OTR) COUNTS]")
    for sc, count in sorted(supercat_counts.items()):
        print(f"  - {sc:15}: {count // 2:6} images")

    print(f"\n[CATEGORY COUNTS (Image Frequency | Total Objects)]")
    for key in sorted(cat_img_counts.keys()):
        name = get_display_name(key).capitalize()
        print(f"  - {name:30}: {cat_img_counts[key] // 2:6} images | {cat_obj_counts[key] // 2:8} objects")

    print(f"\n[SPECIFIC CODE COUNTS (Includes General and Sub-variants)]")
    for key in sorted(subcat_img_counts.keys()):
        name = get_display_name(key).capitalize()
        print(f"  - {name:35}: {subcat_img_counts[key] // 2:6} images | {subcat_obj_counts[key] // 2:8} objects")

    # Sanity Checks
    print(f"\n[INTEGRITY CHECK]")
    print(f"  Sum of category objects:    {sum(cat_obj_counts.values()) // 2}")
    print(f"  Sum of subcategory objects: {sum(subcat_obj_counts.values()) // 2}")
    print(f"  Total verified objects:     {adj_total_objects}")

    print("\n" + "="*40)
    print("Analysis Complete.")

    # 6. Export to JSON
    stats_dir = project_root / "dataset" / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Update global mapping
    mapping_path = stats_dir / "mapping.json"
    mapping_data = {}
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    mapping_data.update(meta_map)
    mapping_data.update(code_to_name)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4, ensure_ascii=False)

    # Save CoCount-train stats
    train_stats = {
        "summary": {"total_images": adj_total_images, "total_objects": adj_total_objects},
        "type_distribution": {k: v // 2 for k, v in type_counts.items()},
        "supercategory_distribution": {k: v // 2 for k, v in supercat_counts.items()},
        "categories": {k: {"images": cat_img_counts[k] // 2, "objects": cat_obj_counts[k] // 2} for k in cat_img_counts},
        "subcategories": {k: {"images": subcat_img_counts[k] // 2, "objects": subcat_obj_counts[k] // 2} for k in subcat_img_counts}
    }
    with open(stats_dir / "cocount_train_stats.json", 'w', encoding='utf-8') as f:
        json.dump(train_stats, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()