import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Paths
    metadata_path = project_root / "dataset" / "pairtally_dataset" / "annotations" / "image_metadata.json"
    manifest_path = project_root / "dataset" / "pairtally_dataset" / "annotations" / "pairtally_annotations_simple.json"
    
    def normalize_name(n):
        """Standardize names for robust mapping: remove parentheses and non-alphanum."""
        n = re.sub(r'\(.*?\)', '', str(n))
        n = re.sub(r'[^\w\s]', ' ', n.lower())
        return " ".join(n.split())

    # 0. Load Metadata Mapping for strict naming
    id_to_name = {}
    if metadata_path.exists():
        print(f"Loading metadata mapping from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

            # The metadata is a dictionary where keys are original image names,
            # and values are dictionaries containing category/attribute info.
            # We need to iterate through the values to get the unique category/attribute definitions.
            # To avoid duplicates, we use a set of processed (super_cat_id, cat_id, attr_id) tuples.
            processed_ids = set() # To track (super_cat_id, cat_id, attr_id) tuples
            for img_key, img_info in meta_data.items(): # Iterate through image entries
                super_c_code = str(img_info.get('super_category_id', '')).strip().upper()
                c_code = str(img_info.get('category_id', '')).strip().upper()
                c_name = str(img_info.get('category', 'Unknown')).strip().lower()
                attr_name = str(img_info.get('attribute', c_name)).strip().lower()
                attr_id = str(img_info.get('category_attribute_id', '')).strip().upper()
                
                # Add simple category mapping (e.g., BEA0 -> bean) for fallback
                if c_code and (None, c_code, '0') not in processed_ids:
                    id_to_name[f"{c_code}0"] = c_name
                    processed_ids.add((None, c_code, '0'))
                # Add combined category mapping (e.g., FOO_BEA0 -> bean)
                if super_c_code and c_code and (super_c_code, c_code, '0') not in processed_ids:
                    id_to_name[f"{super_c_code}_{c_code}0"] = c_name
                    processed_ids.add((super_c_code, c_code, '0'))
                
                # Add simple subcategory mapping (e.g., BEA1 -> black bean) for fallback
                if attr_id and (None, attr_id, '') not in processed_ids:
                    id_to_name[attr_id] = attr_name
                    processed_ids.add((None, attr_id, ''))
                # Add combined subcategory mapping (e.g., FOO_BEA1 -> black bean)
                if super_c_code and attr_id and (super_c_code, attr_id, '') not in processed_ids:
                    id_to_name[f"{super_c_code}_{attr_id}"] = attr_name
                    processed_ids.add((super_c_code, attr_id, ''))

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    # 1. Load Manifest
    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    total_images = 0
    total_objects = 0
    
    type_counts = Counter()
    supercat_counts = Counter()
    cat_img_counts = Counter()
    cat_obj_counts = Counter()
    subcat_img_counts = Counter()
    subcat_obj_counts = Counter()
    
    supercat_map = {"FOO": "Food", "FUN": "Fun", "HOU": "Household", "OFF": "Office", "OTR": "Other"}

    print("Analyzing PairTally manifest...")

    for img_name, data in manifest.items():
        # 1. Extraction with Fallbacks (Root -> Components -> Filename)
        comp = data.get('components', {})
        
        # Type & Supercat
        img_type = (data.get('test_type') or comp.get('test_type') or "Unknown").upper()
        super_code = (data.get('super_category') or comp.get('super_category') or "Unknown").upper()
        
        # Codes (Priority: Root -> Components -> Filename)
        pos_code = (data.get('positive_code') or comp.get('pos_code') or "UNK").upper()
        neg_code = (data.get('negative_code') or comp.get('neg_code') or "UNK").upper()

        # Parse filename key for fallback info
        stem_parts = Path(img_name).stem.upper().split('_')
        if img_type == "UNKNOWN":
            img_type = next((p for p in stem_parts if p in ["INTER", "INTRA"]), "Unknown")
        if super_code == "UNKNOWN":
            super_code = next((p for p in stem_parts if p in supercat_map), "Unknown")
        
        if pos_code == "UNK" or neg_code == "UNK":
            obj_codes = [p for p in stem_parts if len(p) == 4 and p[:3].isalpha() and p[3].isdigit()]
            if pos_code == "UNK" and len(obj_codes) > 0: pos_code = obj_codes[0]
            if neg_code == "UNK" and len(obj_codes) > 1: neg_code = obj_codes[1]

        # Construct unique keys by combining super_code with category/subcategory codes
        cat_key_pos = f"{super_code}_{pos_code[:3]}0" if super_code != "Unknown" else f"{pos_code[:3]}0"
        cat_key_neg = f"{super_code}_{neg_code[:3]}0" if super_code != "Unknown" else f"{neg_code[:3]}0"

        subcat_key_pos = f"{super_code}_{pos_code}" if super_code != "Unknown" else pos_code
        subcat_key_neg = f"{super_code}_{neg_code}" if super_code != "Unknown" else neg_code

        super_name = supercat_map.get(super_code, f"Unknown ({super_code})")
        
        # Prompts (Subcategories)
        pos_sub = data.get('positive_prompt') or comp.get('pos_prompt') or "Unknown"
        neg_sub = data.get('negative_prompt') or comp.get('neg_prompt') or "Unknown"

        # Cache mappings if not in metadata to preserve descriptive names
        if subcat_key_pos not in id_to_name:
            id_to_name[subcat_key_pos] = pos_sub
        if subcat_key_neg not in id_to_name:
            id_to_name[subcat_key_neg] = neg_sub

        # Counts
        pos_n = data.get('positive_count', len(data.get('points', [])))
        neg_n = data.get('negative_count', len(data.get('negative_points', [])))

        total_images += 1
        total_objects += (pos_n + neg_n)
        
        type_counts[img_type] += 1
        supercat_counts[super_name] += 1
        
        cat_img_counts[cat_key_pos] += 1
        cat_obj_counts[cat_key_pos] += pos_n
        if cat_key_pos != cat_key_neg:
            cat_img_counts[cat_key_neg] += 1
        cat_obj_counts[cat_key_neg] += neg_n

        subcat_img_counts[subcat_key_pos] += 1
        subcat_obj_counts[subcat_key_pos] += pos_n
        if subcat_key_pos != subcat_key_neg:
            subcat_img_counts[subcat_key_neg] += 1
        subcat_obj_counts[subcat_key_neg] += neg_n

    # 5. Output Report
    print("\n" + "="*45)
    print(" PAIRTALLY BENCHMARK ANALYSIS")
    print("="*45)

    print(f"\n[BASIC STATS]")
    print(f"Total Unique Images (Scenes): {total_images}")
    print(f"Total Unique Objects:         {total_objects}")
    print(f"Avg Objects/Image:            {total_objects/total_images:.2f}" if total_images > 0 else "N/A")
    print(f"Total Unique Categories:      {len(cat_img_counts)}")
    print(f"Total Unique Subcategories:   {len(subcat_img_counts)}")

    def get_display_name(key):
        # Try exact key (FOO_BEA0)
        if key in id_to_name: return id_to_name[key]
        # Try fallback to simple code (BEA0)
        code_only = key.split('_')[-1] if '_' in key else key
        if code_only in id_to_name: return id_to_name[code_only]
        return key

    print(f"\n[TYPE DISTRIBUTION]")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t:10}: {count:6} images")

    print(f"\n[SUPERCATEGORY (FOO/FUN/HOU/OFF/OTR) COUNTS]")
    for sc, count in sorted(supercat_counts.items()):
        print(f"  - {sc:15}: {count:6} images")

    print(f"\n[CATEGORY COUNTS (Images | Objects)]")
    sorted_cat_keys = sorted(cat_img_counts.keys(), key=lambda k: get_display_name(k).lower())
    for key in sorted_cat_keys:
        name = get_display_name(key).capitalize()
        print(f"  - {name:30}: {cat_img_counts[key]:6} images | {cat_obj_counts[key]:8} objects")

    print(f"\n[SUBCATEGORY COUNTS (Images | Objects)]")
    sorted_subcat_keys = sorted(subcat_img_counts.keys(), key=lambda k: get_display_name(k).lower())
    for key in sorted_subcat_keys:
        name = get_display_name(key).capitalize()
        print(f"  - {name:35}: {subcat_img_counts[key]:6} images | {subcat_obj_counts[key]:8} objects")

    print(f"\n[INTEGRITY CHECK]")
    print(f"  Verified total objects match: {total_objects == sum(subcat_obj_counts.values())}")
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
    mapping_data.update(id_to_name)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4, ensure_ascii=False)

    # Save PairTally stats
    pairtally_stats = {
        "summary": {"total_images": total_images, "total_objects": total_objects},
        "type_distribution": dict(type_counts),
        "supercategory_distribution": dict(supercat_counts),
        "categories": {k: {"images": cat_img_counts[k], "objects": cat_obj_counts[k]} for k in cat_img_counts},
        "subcategories": {k: {"images": subcat_img_counts[k], "objects": subcat_obj_counts[k]} for k in subcat_img_counts}
    }
    with open(stats_dir / "pairtally_stats.json", 'w', encoding='utf-8') as f:
        json.dump(pairtally_stats, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()