import os
import sys
import argparse

def rename_images(folder_path, to_simple=False):
    """
    Rename image files:
      - Default: 0.jpg~xxxx.jpg -> 00000.jpg~0xxxx.jpg (5 digits with leading zeros)
      - to_simple=True: 00000.jpg~0xxxx.jpg -> 0.jpg~xxxx.jpg (remove leading zeros)
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Get all jpg files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # Filter files that match the pattern (numeric name + .jpg)
    image_files = []
    for file in files:
        name_without_ext = file[:-4]  # Remove .jpg extension
        # Accept numbers with or without leading zeros
        if name_without_ext.isdigit():
            image_files.append(file)
    
    # Sort files by numeric value (ignore leading zeros)
    image_files.sort(key=lambda x: int(x[:-4]))
    
    # Rename files
    for file in image_files:
        old_path = os.path.join(folder_path, file)
        name_without_ext = file[:-4]
        if to_simple:
            # Convert 00000.jpg to 0.jpg (remove leading zeros)
            new_name = f"{int(name_without_ext)}.jpg"
        else:
            # Convert 0.jpg to 00000.jpg (5 digits)
            new_name = f"{int(name_without_ext):05d}.jpg"
        new_path = os.path.join(folder_path, new_name)
        
        if old_path != new_path:
            if os.path.exists(new_path):
                print(f"Warning: Skipping {file}, target {new_name} already exists!")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rename images between 0.jpg <-> 00000.jpg format")
    parser.add_argument("folder_path", type=str, help="Path to the image folder")
    parser.add_argument("--to-simple", action="store_true", help="Convert to no-leading-zero format (e.g., 00001.jpg -> 1.jpg)")
    args = parser.parse_args()
    
    rename_images(args.folder_path, to_simple=args.to_simple)

