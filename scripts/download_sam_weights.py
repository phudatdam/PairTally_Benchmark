import os
import urllib.request
from pathlib import Path

def main():
    """
    Downloads the SAM ViT-H checkpoint to the models directory.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / 'models'
    
    # Create models directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # SAM ViT-H is the largest and most accurate model
    filename = 'sam_vit_h_4b8939.pth'
    url = f"https://dl.fbaipublicfiles.com/segment_anything/{filename}"
    destination = model_dir / filename

    if destination.exists():
        print(f"Checkpoint already exists at: {destination}")
        return

    print(f"Downloading {filename} from official Meta AI servers...")
    print("Note: This file is approximately 2.4 GB. This may take several minutes.")

    def progress_report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            print(f"\rProgress: {percent:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="")
        else:
            print(f"\rDownloaded: {downloaded / 1e6:.1f} MB", end="")

    try:
        urllib.request.urlretrieve(url, str(destination), reporthook=progress_report)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading weights: {e}")

if __name__ == "__main__":
    main()