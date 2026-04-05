import os
import json
from pathlib import Path
from datasets import load_dataset
from collections import Counter, defaultdict

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    local_dir = project_root / "dataset" / "CoCount-test"
    local_data_dir = local_dir / "data"

    import re
    def clean_name(n):
        # Remove content in parentheses, non-alphanumeric chars, and normalize spaces
        n = re.sub(r'\(.*?\)', '', n)
        n = re.sub(r'[^\w\s]', '', n.lower())
        return " ".join(n.split())

    # 0. Load Metadata for strict mapping
    metadata_path = project_root / "dataset" / "pairtally_dataset" / "annotations" / "image_metadata.json"
    meta_map = {}
    name_to_code = {} # Reverse mapping for test set robustness
    if metadata_path.exists():
        print(f"Loading metadata mapping from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f) # meta_data is a dict of image_name -> info
            
            processed_ids = set() # To track (super_cat_id, cat_id, attr_id) tuples
            for img_key, img_info in meta_data.items():
                super_c_code = img_info.get('super_category_id', '').strip().upper()
                c_code = img_info.get('category_id', '').strip().upper()
                c_name = clean_name(img_info.get('category', ''))
                attr_name = clean_name(img_info.get('attribute', ''))
                attr_id = img_info.get('category_attribute_id', 'UNK').strip().upper()

                if c_code:
                    simple_cat_key = f"{c_code}0"
                    if (None, c_code, '0') not in processed_ids:
                        meta_map[simple_cat_key] = c_name
                        if c_name: name_to_code[c_name] = simple_cat_key
                        processed_ids.add((None, c_code, '0'))
                    if super_c_code and (super_c_code, c_code, '0') not in processed_ids:
                        combined_cat_key = f"{super_c_code}_{c_code}0"
                        meta_map[combined_cat_key] = c_name
                        if c_name: name_to_code[c_name] = combined_cat_key
                        processed_ids.add((super_c_code, c_code, '0'))
                if attr_id:
                    simple_attr_key = attr_id
                    if (None, attr_id, '') not in processed_ids:
                        meta_map[simple_attr_key] = attr_name
                        if attr_name: name_to_code[attr_name] = simple_attr_key
                        processed_ids.add((None, attr_id, ''))
                    if super_c_code and (super_c_code, attr_id, '') not in processed_ids:
                        combined_attr_key = f"{super_c_code}_{attr_id}"
                        meta_map[combined_attr_key] = attr_name
                        if attr_name: name_to_code[attr_name] = combined_attr_key
                        processed_ids.add((super_c_code, attr_id, ''))

    if not local_data_dir.exists():
        print(f"Error: The directory {local_data_dir} does not exist.")
        return

    parquet_files = [str(f.resolve()) for f in local_data_dir.glob("*.parquet")]
    if not parquet_files:
        print(f"No .parquet files found in {local_data_dir}.")
        return

    print(f"Loading {len(parquet_files)} parquet files from CoCount-test...")
    # The test set usually loads as 'train' or 'test' split depending on how parquet was generated
    ds = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
    
    # Remove the image column to keep processing light
    ds_light = ds.remove_columns(["image"])

    total_samples = 0
    total_objects_raw = 0
    
    type_counts = Counter()        # INTER vs INTRA
    supercat_counts = Counter()    # Food, Fun, etc.
    
    # Stats grouped by Prefix (Category)
    cat_img_counts = Counter()
    cat_obj_counts = Counter()
    
    # Stats grouped by Full Code (Subcategory)
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

    print("Iterating over test dataset...")

    for row in ds_light:
        total_samples += 1
        
        # 1. Extract Captions and Counts
        pos_sub = row.get("pos_caption", "Unknown").strip()
        neg_sub = row.get("neg_caption", "Unknown").strip()
        
        pos_n = row.get("pos_count", 0)
        neg_n = row.get("neg_count", 0)
        total_objects_raw += (pos_n + neg_n)
        
        # 2. Derive Structural Info
        super_code = row.get("category", "Unknown").upper()
        
        # 2. Resolve Codes
        # Priority 1: Extract from video_id (highly reliable in CoCount)
        vid = row.get("video_id", "")
        vid_parts = vid.split('_')
        # Look for 4-char patterns like BEA1 or PAS0
        potential_codes = [p for p in vid_parts if len(p) == 4 and p[:3].isalpha() and p[3].isdigit()]
        
        resolved_pos_code = "UNK?"
        resolved_neg_code = "UNK?"

        if len(potential_codes) >= 2:
            # Prepend super_code to align with unique metadata keys
            code_a = f"{super_code}_{potential_codes[0]}" if super_code != "Unknown" else potential_codes[0]
            code_b = f"{super_code}_{potential_codes[1]}" if super_code != "Unknown" else potential_codes[1]
            # Disambiguate using name_to_code mapping
            match_pos = name_to_code.get(clean_name(pos_sub))
            if match_pos == code_b:
                resolved_pos_code, resolved_neg_code = code_b, code_a
            else:
                resolved_pos_code, resolved_neg_code = code_a, code_b
        else:
            # Priority 2: Fallback to robust caption-to-code matching
            resolved_pos_code = name_to_code.get(clean_name(pos_sub), "UNK?")
            resolved_neg_code = name_to_code.get(clean_name(neg_sub), "UNK?")

        # 3. Derive Structural Info
        super_name = supercat_map.get(super_code, f"Unknown ({super_code})")
        supercat_counts[super_name] += 1

        # Determine INTER/INTRA by comparing prefixes
        # Derive category keys from the resolved subcategory codes (e.g., FOO_BEA1 -> FOO_BEA0)
        cat_key_pos = (resolved_pos_code[:-1] + '0') if resolved_pos_code != "UNK?" else "UNK?0"
        cat_key_neg = (resolved_neg_code[:-1] + '0') if resolved_neg_code != "UNK?" else "UNK?0"
        
        # Logic: use video_id prefix for TYPE if available, otherwise compare prefixes
        if vid.startswith("INTER"):
            img_type = "INTER"
        elif vid.startswith("INTRA"):
            img_type = "INTRA"
        else: # Compare the category parts of the combined keys
            img_type = "INTRA" if cat_key_pos == cat_key_neg and cat_key_pos != "UNK?0" else "INTER"
            
        type_counts[img_type] += 1

        def resolve_name(key, default_name):
            # Try full key (FOO_BEA1)
            if key in meta_map: return meta_map[key]
            # Try fallback to simple code (BEA1)
            code_only = key.split('_')[-1] if '_' in key else key
            if code_only in meta_map: return meta_map[code_only]
            return default_name

        # Cache mappings
        if resolved_pos_code not in code_to_name:
            code_to_name[resolved_pos_code] = resolve_name(resolved_pos_code, pos_sub)
        if resolved_neg_code not in code_to_name:
            code_to_name[resolved_neg_code] = resolve_name(resolved_neg_code, neg_sub)

        # 4. Aggregation
        # Category stats (Combined Prefixes)
        cat_img_counts[cat_key_pos] += 1
        cat_obj_counts[cat_key_pos] += pos_n
        if cat_key_pos != cat_key_neg:
            cat_img_counts[cat_key_neg] += 1
        cat_obj_counts[cat_key_neg] += neg_n

        # Subcategory stats (Full Combined Codes)
        subcat_img_counts[resolved_pos_code] += 1
        subcat_obj_counts[resolved_pos_code] += pos_n
        if resolved_pos_code != resolved_neg_code:
            subcat_img_counts[resolved_neg_code] += 1
        subcat_obj_counts[resolved_neg_code] += neg_n

    # 5. Output Results
    print("\n" + "="*45)
    print(" COCOUNT-TEST DATASET ANALYSIS")
    print("="*45)
    
    # Adjustment for paired samples
    adj_total_images = total_samples // 2
    adj_total_objects = total_objects_raw // 2

    all_codes = list(subcat_img_counts.keys())
    general_codes = [c for c in all_codes if c.endswith('0')]
    actual_sub_codes = [c for c in all_codes if not c.endswith('0')]

    print(f"\n[BASIC STATS (Adjusted for Paired Scenes)]")
    print(f"Total Unique Images (Pairs): {adj_total_images}")
    print(f"Total Unique Objects:        {adj_total_objects}")
    print(f"Avg Objects/Image:           {adj_total_objects/adj_total_images:.2f}" if adj_total_images > 0 else "N/A")
    print(f"Total Unique Category Groups: {len(cat_img_counts)}")
    print(f"Total Unique Codes:           {len(all_codes)}")
    print(f"  - Base Categories (suffix 0):    {len(general_codes)}")
    print(f"  - Specific Sub-variants (1/2):   {len(actual_sub_codes)}")

    print(f"\n[CODE TO NAME MAPPING (Grouped by Prefix)]")
    grouped_mappings = defaultdict(dict)
    for code, name in code_to_name.items():
        parts = code.split('_')
        if len(parts) == 2: # e.g., FOO_BEA1
            super_prefix = parts[0]
            sub_code_part = parts[1] # BEA1
            prefix = sub_code_part[:3] # BEA
            combined_prefix_key = f"{super_prefix}_{prefix}" # FOO_BEA
            grouped_mappings[combined_prefix_key][code] = name
        else: # Fallback for non-combined codes (e.g., just "BEA1" if supercat was "Unknown")
            prefix = code[:3]
            grouped_mappings[prefix][code] = name # Key: "BEA", Value: { "BEA1": "black bean" }

    def get_display_name(key):
        # Check meta_map first (Metadata is ground truth)
        if key in meta_map: return meta_map[key]
        # Fallback to simple code in meta_map
        code_only = key.split('_')[-1] if '_' in key else key
        if code_only in meta_map: return meta_map[code_only]
        # Fallback to code_to_name (manifest captions)
        return code_to_name.get(key, key)

    for combined_prefix_key in sorted(grouped_mappings.keys()): # combined_prefix_key is like FOO_BEA or just BEA
        m = grouped_mappings[combined_prefix_key]
        super_prefix = ""
        prefix = combined_prefix_key
        if '_' in combined_prefix_key:
            super_prefix, prefix = combined_prefix_key.split('_')
        
        cat = get_display_name(f"{super_prefix}_{prefix}0" if super_prefix else f"{prefix}0").capitalize()
        sub1 = get_display_name(f"{super_prefix}_{prefix}1" if super_prefix else f"{prefix}1").capitalize()
        sub2 = get_display_name(f"{super_prefix}_{prefix}2" if super_prefix else f"{prefix}2").capitalize()
        
        print(f"  {combined_prefix_key:8} | Cat: {cat:30} | Sub1: {sub1:30} | Sub2: {sub2}")

    print(f"\n[TYPE DISTRIBUTION]")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t:10}: {count // 2:6} images")

    print(f"\n[SUPERCATEGORY COUNTS]")
    for sc, count in sorted(supercat_counts.items()):
        print(f"  - {sc:15}: {count // 2:6} images")

    print(f"\n[CATEGORY COUNTS (Unique Images | Actual Objects)]")
    for key in sorted(cat_img_counts.keys()):
        name = get_display_name(key).capitalize()
        print(f"  - {name:30}: {cat_img_counts[key] // 2:6} images | {cat_obj_counts[key] // 2:8} objects")

    print(f"\n[SPECIFIC CODE COUNTS (Unique Images | Actual Objects)]")
    for key in sorted(subcat_img_counts.keys()):
        name = get_display_name(key).capitalize()
        print(f"  - {name:35}: {subcat_img_counts[key] // 2:6} images | {subcat_obj_counts[key] // 2:8} objects")

    print(f"\n[INTEGRITY CHECK]")
    print(f"  Verified total objects match: {adj_total_objects == (sum(cat_obj_counts.values()) // 2)}")
    print("\n" + "="*45)

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

    # Save CoCount-test stats
    test_stats = {
        "summary": {"total_images": adj_total_images, "total_objects": adj_total_objects},
        "type_distribution": {k: v // 2 for k, v in type_counts.items()},
        "supercategory_distribution": {k: v // 2 for k, v in supercat_counts.items()},
        "categories": {k: {"images": cat_img_counts[k] // 2, "objects": cat_obj_counts[k] // 2} for k in cat_img_counts},
        "subcategories": {k: {"images": subcat_img_counts[k] // 2, "objects": subcat_obj_counts[k] // 2} for k in subcat_img_counts}
    }
    with open(stats_dir / "cocount_test_stats.json", 'w', encoding='utf-8') as f:
        json.dump(test_stats, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()