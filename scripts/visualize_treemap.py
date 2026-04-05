import json
import pandas as pd
import plotly.express as px
from pathlib import Path

def get_display_name(key, mapping):
    """Replicates the fallback logic used in the analysis scripts."""
    # Try exact key (e.g., FOO_BEA1)
    if key in mapping:
        return mapping[key].capitalize()
    
    # Try fallback to simple code (e.g., BEA1)
    code_only = key.split('_')[-1] if '_' in key else key
    if code_only in mapping:
        return mapping[code_only].capitalize()
    
    return key.capitalize()

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    stats_dir = project_root / "dataset" / "statistics"
    mapping_path = stats_dir / "mapping.json"

    if not mapping_path.exists():
        print(f"Error: Mapping file not found at {mapping_path}")
        return

    # 1. Load Mapping
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    supercat_map = {
        "FOO": "Food", "FUN": "Fun", "HOU": "Household",
        "OFF": "Office", "OTR": "Other"
    }

    # 2. Select Stats File
    stats_files = list(stats_dir.glob("*_stats.json"))
    if not stats_files:
        print(f"No statistics files found in {stats_dir}")
        return

    print("\nSelect a dataset to visualize (Treemap):")
    for i, f in enumerate(stats_files):
        print(f"{i+1}. {f.name}")
    
    try:
        choice = int(input("\nEnter number: ")) - 1
        selected_file = stats_files[choice]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    with open(selected_file, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)

    # 3. Process Hierarchy
    # We use 'subcategories' as the leaf nodes to build the full tree
    rows = []
    for full_code, counts in stats_data.get("subcategories", {}).items():
        # Split code into parts (e.g., FOO_BEA1)
        parts = full_code.split('_')
        if len(parts) == 2:
            super_code = parts[0]
            sub_code = parts[1]
        else:
            super_code = "Unknown"
            sub_code = full_code

        # Derive Category Key (e.g., FOO_BEA0)
        cat_prefix = sub_code[:3]
        cat_key = f"{super_code}_{cat_prefix}0" if super_code != "Unknown" else f"{cat_prefix}0"

        super_name = supercat_map.get(super_code, super_code)
        cat_name = get_display_name(cat_key, mapping)
        sub_name = get_display_name(full_code, mapping)

        # Avoid redundancy if the subcategory name is the same as the category name
        if sub_name == cat_name:
            sub_name = f"{sub_name} (General)"

        rows.append({
            "Supercategory": super_name,
            "Category": cat_name,
            "Subcategory": sub_name,
            "Images": counts["images"],
            "Objects": counts["objects"]
        })

    df = pd.DataFrame(rows)

    # 4. Create Treemap
    fig = px.treemap(
        df,
        path=['Supercategory', 'Category', 'Subcategory'],
        values='Objects',
        title=f"Treemap of Objects: {selected_file.stem.replace('_', ' ').title()}",
        hover_data=['Images'],
        color='Supercategory',
        color_discrete_map={v: px.colors.qualitative.Safe[i] for i, v in enumerate(supercat_map.values())}
    )

    fig.update_traces(textinfo="label+value")
    print(f"Opening treemap diagram for {selected_file.name}...")
    fig.show()

if __name__ == "__main__":
    main()