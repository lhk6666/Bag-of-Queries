import os
import sys

def rename_images(folder_path):
    """
    Rename image files from format 0.jpg~xxxx.jpg to 00000.jpg~0xxxx.jpg
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
        if name_without_ext.isdigit():
            image_files.append(file)
    
    # Sort files by numeric value
    image_files.sort(key=lambda x: int(x[:-4]))
    
    # Rename files
    for file in image_files:
        old_path = os.path.join(folder_path, file)
        name_without_ext = file[:-4]
        new_name = f"{int(name_without_ext):05d}.jpg"  # Format to 5 digits with leading zeros
        new_path = os.path.join(folder_path, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {file} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_rename.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    rename_images(folder_path)