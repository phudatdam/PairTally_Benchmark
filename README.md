# PairTally Bounding Box Annotator

This repository provides a GUI tool for annotating bounding boxes in the PairTally dataset.

## Data Setup

1. **Download Data**: Obtain the PairTally images and initial metadata from [Google Drive](https://drive.google.com/drive/folders/1yaDq87l-Ha08OjgIFwKS7GluzGekah0t?usp=drive_link).
2. **Organize Directory**: Extract the downloaded data so it follows the structure below.

## Expected File Structure

```
PairTally_Benchmark/
├── dataset/
│ ├── CoCount-train/
│ │ ├── CoCount-train-raw/ # Raw CoCount train parquet files
│ │ ├── processed_dataset/ # Processed CoCount train dataset
│ │ │ ├── Anno/
│ │ │ └── Image/
│ │ └── weird_bbox.txt # Log of images with example bounding boxes that don't fully contain the object
│ │ └── Anno_with_exam_bbox/ # Temporary folder for SAM output with scores
│ ├── PairTally/
│ │ ├── pairtally_dataset/ # Original PairTally dataset
│ │ ├── processed_dataset/ # Processed PairTally dataset
│ │ │ ├── Anno/
│ │ │ ├── Image/
│ │ │ └── mask/
│ │ ├── removed/ # Folder for filtered-out images
│ │ ├── removed.txt # Log of removed filenames
│ │ └── weird_bbox.txt # Log of images with example bounding boxes that don't fully contain the object
│ └── statistics/ # JSON analysis results and mappings
├── scripts/
│   ├── annotate_bboxes.py    # Main annotation interface
│   ├── visualize_masks.py    # Script for visualizing segmentation masks
│   ├── visualize_cocount_raw.py # Browse raw CoCount images and exemplars
│   ├── sync_cocount_points.py # Sync points from raw CoCount to processed annotations
│   └── update_exam_bboxes_with_sam_scores.py # Update exam_bbox with top 10 SAM predictions
│   ├── filter_exam_bboxes.py  # Interactive tool to refine and replace exam bboxes
└── README.md
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install Pillow
   ```
2. **Run Script**:
   From the project root directory, run:
   ```bash
   python scripts/annotate_bboxes.py
   ```

Detailed usage instructions and keyboard shortcuts are available via the **Help** button within the application.