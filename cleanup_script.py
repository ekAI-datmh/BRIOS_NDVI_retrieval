import os
import shutil

def remove_subfolders(base_data_path, folders_to_remove):
    """Removes specified subfolders from each directory within the base_data_path."""
    
    print(f"Starting cleanup in: {base_data_path}")
    print(f"Folders to remove if found: {', '.join(folders_to_remove)}\n")
    
    # Check if the base path exists
    if not os.path.isdir(base_data_path):
        print(f"Error: Base data path not found or is not a directory: {base_data_path}")
        return

    # Iterate through ROI directories
    for roi_dirname in os.listdir(base_data_path):
        roi_path = os.path.join(base_data_path, roi_dirname)
        if not os.path.isdir(roi_path):
            print(f"Skipping non-directory item: {roi_dirname}")
            continue # Skip files, only process directories
            
        print(f"Processing ROI: {roi_dirname}")
        
        # Iterate through the subfolders to remove
        for folder_name in folders_to_remove:
            subfolder_path = os.path.join(roi_path, folder_name)
            
            if os.path.isdir(subfolder_path):
                try:
                    shutil.rmtree(subfolder_path)
                    print(f"  Removed: {subfolder_path}")
                except OSError as e:
                    print(f"  Error removing {subfolder_path}: {e}")
            else:
                print(f"  Not found (or not a directory): {subfolder_path}")
                
        print("---")
        
    print("Cleanup process finished.")

# --- Configuration ---
# Use the absolute path provided in the context
BASE_DOWNLOAD_PATH = "/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_crawled"
SUBFOLDERS_TO_REMOVE = ["era5", "modis", "lst"]

# --- Run the cleanup ---
# Make sure you want to do this before uncommenting and running!
remove_subfolders(BASE_DOWNLOAD_PATH, SUBFOLDERS_TO_REMOVE)

# Safety check: Print what would be done instead of actually doing it first.
print("--- DRY RUN --- ")
print("The script would attempt to remove the following folders if uncommented:")

if not os.path.isdir(BASE_DOWNLOAD_PATH):
    print(f"Error: Base data path not found or is not a directory: {BASE_DOWNLOAD_PATH}")
else:
    for roi_dirname in os.listdir(BASE_DOWNLOAD_PATH):
        roi_path = os.path.join(BASE_DOWNLOAD_PATH, roi_dirname)
        if not os.path.isdir(roi_path):
            continue
            
        for folder_name in SUBFOLDERS_TO_REMOVE:
            subfolder_path = os.path.join(roi_path, folder_name)
            if os.path.isdir(subfolder_path):
                 print(f"DELETE TARGET: {subfolder_path}")

print("\nIf this looks correct, uncomment the line `remove_subfolders(...)` above and run the script.") 