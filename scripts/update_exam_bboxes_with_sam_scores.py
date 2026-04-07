import json
from pathlib import Path

def main():
    """
    Updates the 'exam_bbox' field in JSON annotation files in
    'dataset/CoCount-train/processed_dataset/Anno' with the top 10
    highest-scoring bounding boxes from corresponding files in
    'dataset/CoCount-train/processed_dataset/Anno_with_exam_bbox'.
    Additionally removes the 'points' field from all target annotation files.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Source directory containing JSONs with SAM scores
    source_anno_dir = project_root / "dataset" / "CoCount-train" / "processed_dataset" / "Anno_with_exam_bbox"
    # Target directory to update
    target_anno_dir = project_root / "dataset" / "CoCount-train" / "processed_dataset" / "Anno"

    if not source_anno_dir.exists():
        print(f"Error: Source annotation directory not found at {source_anno_dir}")
        return
    if not target_anno_dir.exists():
        print(f"Error: Target annotation directory not found at {target_anno_dir}")
        return

    print(f"Scanning for JSON files in {target_anno_dir}...")
    target_json_files = list(target_anno_dir.glob("*.json"))

    if not target_json_files:
        print(f"No JSON files found in {target_anno_dir}.")
        return

    updated_count = 0
    points_removed_count = 0

    for target_json_path in target_json_files:
        source_json_path = source_anno_dir / target_json_path.name

        try:
            with open(target_json_path, 'r', encoding='utf-8') as f:
                target_data = json.load(f)

            modified = False
            if 'points' in target_data:
                del target_data['points']
                points_removed_count += 1
                modified = True

            if source_json_path.exists():
                with open(source_json_path, 'r', encoding='utf-8') as f_source:
                    source_data = json.load(f_source)
                
                # Extract exam_bbox with scores
                exam_bboxes_with_scores = source_data.get('exam_bbox', [])
                
                # Sort by score in descending order and take the top 10
                sorted_bboxes = sorted(exam_bboxes_with_scores, key=lambda x: x.get('score', 0), reverse=True)
                top_10_bboxes = [item['bbox'] for item in sorted_bboxes[:10]]

                target_data['exam_bbox'] = top_10_bboxes
                updated_count += 1
                modified = True

            if modified:
                with open(target_json_path, 'w', encoding='utf-8') as f:
                    json.dump(target_data, f, indent=4)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {target_json_path.name}: {e}")
        except IOError as e:
            print(f"File I/O error for {target_json_path.name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {target_json_path.name}: {e}")

    print(f"\nProcessing complete.")
    print(f"Updated 'exam_bbox' in {updated_count} files.")
    print(f"Removed 'points' field from {points_removed_count} files.")

if __name__ == "__main__":
    main()