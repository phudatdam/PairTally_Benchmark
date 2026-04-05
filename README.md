# PairTally Bounding Box Annotator

This repository provides a GUI tool for annotating bounding boxes in the PairTally dataset.

## Data Setup

1. **Download Data**: Obtain the PairTally images and initial metadata from [Google Drive](https://drive.google.com/drive/folders/1yaDq87l-Ha08OjgIFwKS7GluzGekah0t?usp=drive_link).
2. **Organize Directory**: Extract the downloaded data so it follows the structure below.

## Expected File Structure

```
PairTally_Benchmark/
├── dataset/
│   ├── processed_dataset/
│   │   ├── Anno/             # Active JSON annotation files
│   │   ├── Image/            # Active image files
│   |   └── mask/             # Example segmentation masks
│   ├── removed/
│   │   ├── Anno/             # Annotations for removed images
│   │   └── Image/            # Removed image files
│   ├── removed.txt           # Log of filenames removed from the active set
│   └── weird_bbox.txt        # Log of images with problematic existing objects
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