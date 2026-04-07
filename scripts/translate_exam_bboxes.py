import json
import os
import sys

def convert_box(points):
    """
    Converts a list of 4 points [[x,y], [x,y], [x,y], [x,y]] 
    to a bounding box [xmin, ymin, xmax, ymax].
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]

def main():
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    source_json_path = os.path.join(project_root, 'dataset', 'PairTally', 'pairtally_dataset', 'annotations', 'pairtally_annotations_simple.json')
    
    # We need to check both the active and removed annotation folders
    anno_dirs = [
        os.path.join(project_root, 'dataset', 'PairTally', 'processed_dataset', 'Anno'),
        os.path.join(project_root, 'dataset', 'PairTally', 'removed', 'Anno')
    ]

    print(f"Reading source from: {source_json_path}")

    if not os.path.exists(source_json_path):
        print("Source JSON file does not exist.")
        return

    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except Exception as e:
        print(f"Failed to load source JSON: {e}")
        return

    updated_count = 0

    for img_name, data in source_data.items():
        base_name = os.path.splitext(img_name)[0]
        
        # Extract and convert positive boxes
        pos_boxes_raw = data.get('box_examples_coordinates', [])
        pos_exam_bboxes = [convert_box(box) for box in pos_boxes_raw]

        # Extract and convert negative boxes
        # Note: using 'negative_box_exemples_coordinates' based on original file structure
        neg_boxes_raw = data.get('negative_box_exemples_coordinates', [])
        neg_exam_bboxes = [convert_box(box) for box in neg_boxes_raw]

        # Update positive annotation file
        pos_file_name = f"{base_name}_positive.json"
        for a_dir in anno_dirs:
            pos_file_path = os.path.join(a_dir, pos_file_name)
            if os.path.exists(pos_file_path):
                try:
                    with open(pos_file_path, 'r', encoding='utf-8') as f:
                        anno_data = json.load(f)
                    anno_data['exam_bbox'] = pos_exam_bboxes
                    with open(pos_file_path, 'w', encoding='utf-8') as f:
                        json.dump(anno_data, f, indent=4)
                    updated_count += 1
                except Exception as e:
                    print(f"Error updating {pos_file_name}: {e}")

        # Update negative annotation file
        neg_file_name = f"{base_name}_negative.json"
        for a_dir in anno_dirs:
            neg_file_path = os.path.join(a_dir, neg_file_name)
            if os.path.exists(neg_file_path):
                try:
                    with open(neg_file_path, 'r', encoding='utf-8') as f:
                        anno_data = json.load(f)
                    anno_data['exam_bbox'] = neg_exam_bboxes
                    with open(neg_file_path, 'w', encoding='utf-8') as f:
                        json.dump(anno_data, f, indent=4)
                    updated_count += 1
                except Exception as e:
                    print(f"Error updating {neg_file_name}: {e}")

    print(f"Successfully updated {updated_count} annotation files.")

if __name__ == "__main__":
    main()