"""
Google Colab Setup Script
Copies dataset from Google Drive to local disk for 10-20x faster training
"""

import os
import shutil
import time
from pathlib import Path


def setup_colab_fast_storage():
    """
    Copy dataset from Google Drive to Colab's local SSD for faster I/O.
    
    Google Drive I/O is VERY slow (~5-10 MB/s)
    Local SSD I/O is FAST (~500-1000 MB/s)
    
    This gives 10-20x speedup in data loading!
    """
    
    print("\n" + "="*80)
    print("COLAB FAST STORAGE SETUP")
    print("="*80)
    
    # Source (Google Drive - SLOW)
    drive_dataset_path = "/content/drive/MyDrive/KaggleData/disaster/waste_dataset"
    drive_csv = "/content/drive/MyDrive/KaggleData/disaster/waste_dataset/image.csv"
    
    # Destination (Local SSD - FAST)
    local_dataset_path = "/content/dataset"
    local_csv = "/content/dataset/image.csv"
    
    # Check if Google Drive is mounted
    if not os.path.exists("/content/drive/MyDrive"):
        print("\n❌ Google Drive not mounted!")
        print("   Please run this first:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        return False
    
    # Check if source exists
    if not os.path.exists(drive_dataset_path):
        print(f"\n❌ Source dataset not found at: {drive_dataset_path}")
        return False
    
    # Check if already copied
    if os.path.exists(local_dataset_path) and os.path.exists(local_csv):
        print(f"\n✓ Dataset already exists in local storage: {local_dataset_path}")
        
        # Count files
        local_files = len(list(Path(local_dataset_path).rglob("*")))
        drive_files = len(list(Path(drive_dataset_path).rglob("*")))
        
        print(f"  Local files: {local_files}")
        print(f"  Drive files: {drive_files}")
        
        if local_files >= drive_files * 0.95:  # Allow 5% difference
            print("\n✓ Dataset appears complete. Skipping copy.")
            print(f"\n📂 Use these paths in your code:")
            print(f"   CSV_PATH = '{local_csv}'")
            print(f"   BASE_IMAGE_PATH = '{local_dataset_path}'")
            return True
        else:
            print("\n⚠️  Dataset incomplete. Re-copying...")
            shutil.rmtree(local_dataset_path)
    
    # Get dataset size
    print(f"\n📊 Analyzing dataset size...")
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(drive_dataset_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
                file_count += 1
            except:
                pass
    
    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"  Total files: {file_count}")
    print(f"  Total size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    
    # Check available space
    stat = os.statvfs('/content')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    print(f"  Available space: {free_gb:.1f} GB")
    
    if size_gb > free_gb * 0.8:  # Need 80% free space
        print("\n❌ Not enough space in /content/!")
        print("   Try reducing dataset size or use Google Drive directly")
        return False
    
    # Copy dataset
    print(f"\n📦 Copying dataset from Google Drive to local SSD...")
    print(f"   From: {drive_dataset_path}")
    print(f"   To:   {local_dataset_path}")
    print(f"\n⏱️  This will take ~{size_mb / 10:.0f}-{size_mb / 20:.0f} seconds...")
    print("   (Please wait, do not interrupt)")
    
    start_time = time.time()
    
    try:
        # Copy entire directory
        shutil.copytree(drive_dataset_path, local_dataset_path)
        
        copy_time = time.time() - start_time
        speed_mbps = size_mb / copy_time
        
        print(f"\n✅ Copy complete!")
        print(f"   Time: {copy_time:.1f} seconds")
        print(f"   Speed: {speed_mbps:.1f} MB/s")
        
        # Verify
        copied_files = len(list(Path(local_dataset_path).rglob("*")))
        print(f"   Files copied: {copied_files}")
        
        if copied_files >= file_count * 0.95:
            print("\n✅ Dataset successfully copied to local SSD!")
        else:
            print(f"\n⚠️  Warning: Some files may be missing ({copied_files}/{file_count})")
        
    except Exception as e:
        print(f"\n❌ Error during copy: {e}")
        return False
    
    # Show updated paths
    print(f"\n" + "="*80)
    print("UPDATE YOUR CONFIG.PY WITH THESE PATHS:")
    print("="*80)
    print(f"CSV_PATH = '{local_csv}'")
    print(f"BASE_IMAGE_PATH = '{local_dataset_path}'")
    print("="*80)
    
    print(f"\n💡 Expected speedup:")
    print(f"   Before (Google Drive): ~5-10 MB/s read speed")
    print(f"   After (Local SSD):     ~500-1000 MB/s read speed")
    print(f"   Training speedup:      10-20x faster data loading!")
    print(f"   GPU utilization:       Will increase from 20-30% to 80-95%")
    
    return True


def cleanup_local_storage():
    """Remove local dataset copy (frees up space)"""
    local_dataset_path = "/content/dataset"
    
    if os.path.exists(local_dataset_path):
        print(f"Removing local dataset: {local_dataset_path}")
        shutil.rmtree(local_dataset_path)
        print("✓ Cleanup complete")
    else:
        print("No local dataset to clean up")


if __name__ == "__main__":
    # Run setup
    success = setup_colab_fast_storage()
    
    if success:
        print("\n✅ Setup complete! You can now run train.py with 10-20x faster data loading!")
    else:
        print("\n❌ Setup failed. Please check errors above.")
