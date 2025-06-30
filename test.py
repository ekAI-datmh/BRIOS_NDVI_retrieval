import os
import rasterio
import numpy as np
from tqdm import tqdm
import warnings

# Suppress Rasterio warnings if needed (e.g., NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

def calculate_valid_pixel_percentage(tif_path):
    """
    Calculates the percentage of non-NaN pixels in a GeoTIFF file.

    Args:
        tif_path (str): The path to the GeoTIFF file.

    Returns:
        float: The percentage of valid (non-NaN) pixels, or 0.0 if the file cannot be read.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read the first band
            band1 = src.read(1)
            
            # Get the total number of pixels
            total_pixels = band1.size
            if total_pixels == 0:
                return 0.0

            # Count the number of non-NaN pixels
            # np.isnan() works on float arrays, which is typical for LST data with NaNs.
            non_nan_pixels = np.count_nonzero(~np.isnan(band1))
            
            return (non_nan_pixels / total_pixels) * 100

    except Exception as e:
        print(f"Warning: Could not process file {os.path.basename(tif_path)}. Error: {e}")
        return 0.0

def find_lst_subfolder(roi_path):
    """
    Finds the LST data subfolder within a given ROI directory.
    The LST folder is identified by having 'lst' in its name.

    Args:
        roi_path (str): The path to the ROI's main folder.

    Returns:
        str or None: The full path to the LST folder, or None if not found.
    """
    if not os.path.isdir(roi_path):
        return None
        
    for item in os.listdir(roi_path):
        if "lst" in item.lower() and os.path.isdir(os.path.join(roi_path, item)):
            return os.path.join(roi_path, item)
            
    return None

def analyze_lst_data_validity(big_folder_path):
    """
    Analyzes all ROIs in a directory, scores them based on the validity
    of their LST data, and returns a sorted list of ROIs and their scores.

    Args:
        big_folder_path (str): The path to the main directory containing all ROI folders.

    Returns:
        list: A list of tuples, where each tuple is (roi_name, score),
              sorted in descending order of score. Returns an empty list on error.
    """
    if not os.path.isdir(big_folder_path):
        print(f"Error: The specified folder does not exist: {big_folder_path}")
        return []

    roi_scores = {}
    
    # Get all items in the big folder that are directories
    roi_folders = [d for d in os.listdir(big_folder_path) if os.path.isdir(os.path.join(big_folder_path, d))]
    
    if not roi_folders:
        print(f"Error: No ROI sub-folders found in {big_folder_path}")
        return []

    print(f"Found {len(roi_folders)} potential ROI folders. Starting analysis...")

    for roi_name in tqdm(roi_folders, desc="Processing ROIs"):
        roi_path = os.path.join(big_folder_path, roi_name)
        
        # Find the sub-folder containing LST data
        lst_folder_path = find_lst_subfolder(roi_path)

        if not lst_folder_path:
            # tqdm.write(f"Info: No LST sub-folder found for ROI: {roi_name}. Skipping.")
            continue

        total_validity_score = 0.0
        
        # Get all TIFF files in the LST folder
        tif_files = [f for f in os.listdir(lst_folder_path) if f.lower().endswith(('.tif', '.tiff'))]

        if not tif_files:
            # tqdm.write(f"Info: No .tif files found in LST folder for ROI: {roi_name}.")
            continue
            
        for tif_file in tif_files:
            tif_path = os.path.join(lst_folder_path, tif_file)
            validity_percentage = calculate_valid_pixel_percentage(tif_path)
            total_validity_score += validity_percentage

        roi_scores[roi_name] = total_validity_score

    # Sort the ROIs by their score in descending order
    sorted_rois = sorted(roi_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_rois


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Set the path to your main data folder here
    # This folder should contain the individual ROI folders (e.g., 220kV_Box1, 220kV_Box2, etc.)
    BIG_FOLDER = "/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_retrieval/download_data/"
    
    # The function now returns the list of ROIs and scores
    top_rois = analyze_lst_data_validity(BIG_FOLDER)
    
    if top_rois:
        print("\n--- Top 20 Most Valid LST Data ROIs (as a list of tuples) ---")
        # We get the full list back, here we slice the top 20 for display
        result_list = top_rois[:20]
        print(result_list)
        
        # You can also iterate through it for different formatting
        print("\n--- Formatted Top 20 List ---")
        for i, (roi, score) in enumerate(result_list):
            print(f"#{i+1}: ROI='{roi}', Score={score:.2f}")

    else:
        print("\nNo valid ROIs found or an error occurred.")
