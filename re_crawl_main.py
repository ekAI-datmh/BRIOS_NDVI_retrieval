import ee
import os
import rasterio
from rasterio.warp import transform_bounds
from lst_retrieval import lst_retrive
from rvi_retrieval import main_rvi
from ndvi_retrieval import main_ndvi


def get_region_coordinates(tif_file_path):
    """
    Opens a GeoTIFF file and returns the coordinates of its bounds as a list of
    [x, y] pairs in the EPSG:3857 system. Coordinates are ordered as:
    [top-right, bottom-right, bottom-left, top-left, top-right] (closing the polygon).
    """
    with rasterio.open(tif_file_path) as dataset:
        bounds = dataset.bounds  # (left, bottom, right, top)
        # Transform bounds if dataset's CRS is not EPSG:3857.
        if dataset.crs.to_string() != 'EPSG:3857':
            bounds = transform_bounds(dataset.crs, 'EPSG:3857', *bounds)
        
        # Create coordinates list in the specified order.
        coordinates = [
            [bounds[2], bounds[1]],  # Top-right (right, bottom)
            [bounds[2], bounds[3]],  # Bottom-right (right, top)
            [bounds[0], bounds[3]],  # Bottom-left (left, top)
            [bounds[0], bounds[1]],  # Top-left (left, bottom)
            [bounds[2], bounds[1]]   # Closing the loop (same as first point)
        ]
        return coordinates

def read_region_coordinates(folder_path):
    """
    Reads all TIFF images in the given folder (ignoring non-TIF files), extracts the region name
    from the file name, gets the image bounds, and returns a dictionary with keys as region names
    and values as the coordinates (in EPSG:3857) of that image.
    
    Assumes file names have the format:
    "RegionName_lst16days_YYYY-MM-DD.tif"
    
    For example, for "Giao_Lac_lst16days_2022-12-20.tif", the region name is "Giao_Lac".
    """
    region_dict = {}
    # Loop over all files in the folder.
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".tif"):
            # Extract region name by splitting on "_lst16days"
            region_name = filename.split("_lst16days")[0].replace("_", "") + "_DucCo_GiaLai"
            print(region_name)
            tif_file_path = os.path.join(folder_path, filename)
            try:
                coordinates = get_region_coordinates(tif_file_path)
                region_dict[region_name] = coordinates
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return region_dict

# Example usage:
# folder_path = "/mnt/data1tb/LSTRetrieval/Code/LST"  # Replace with your folder path
# regions = read_region_coordinates(folder_path)
# print(regions)


if __name__=="__main__":
    # Initialize the Earth Engine API.
    ee.Authenticate(force = True)
    ee.Initialize(project='ee-hadat')

    date_start = '2022-12-15'
    date_end = '2025-01-01'
    start_date = ee.Date(date_start)    
    end_date = ee.Date(date_end)
    big_folder = "/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_retrieval/new_download_data"
    roi_regions_folder = "/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_retrieval/LST_DucCo"
    roi_regions = read_region_coordinates(roi_regions_folder)

    for roi_region in list(roi_regions.keys())[:]:
        roi_name, roi = roi_region, roi_regions[roi_region]
        geometry = ee.Geometry.Polygon(
        [roi],
        proj='EPSG:3857'
        )
        print(roi_name)
        lst_retrive(date_start, date_end, geometry, roi_name, big_folder)
        main_rvi(start_date, end_date, geometry, big_folder, roi_name)
        main_ndvi(start_date, end_date, geometry, roi_name, big_folder)
    
    


