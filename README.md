# PairTally Bounding Box Annotator

This repository provides a GUI tool for annotating bounding boxes in the PairTally and CoCount datasets.

## Data Setup

1. **Download Data**: Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1yaDq87l-Ha08OjgIFwKS7GluzGekah0t?usp=drive_link).
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
│ └── PairTally/
│   ├── pairtally_dataset/ # Original PairTally dataset
│   ├── processed_dataset/ # Processed PairTally dataset
│   │ ├── Anno/
│   │ ├── Image/
│   │ └── mask/
│   ├── removed/ # Folder for filtered-out images
│   ├── removed.txt # Log of removed filenames
│   └── weird_bbox.txt # Log of images with example bounding boxes that don't fully contain the object
├── scripts/
│   ├── annotate_bboxes.py    # Main annotation interface
│   └── visualize_masks.py    # Script for visualizing segmentation masks
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