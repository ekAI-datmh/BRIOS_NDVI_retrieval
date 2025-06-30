import ee
import os
import rasterio
from rasterio.warp import transform_bounds
from lst_retrieval import lst_retrive
from rvi_retrieval import main_rvi
from ndvi_retrieval import main_ndvi
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from queue import Queue
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_retrieval.log')
    ]
)

# Rate limiting configuration
class RateLimiter:
    def __init__(self, max_requests_per_minute=30):
        self.max_requests = max_requests_per_minute
        self.requests = Queue()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            while not self.requests.empty():
                if now - self.requests.queue[0] > 60:
                    self.requests.get()
                else:
                    break
            
            # If we're at the limit, wait
            if self.requests.qsize() >= self.max_requests:
                sleep_time = 60 - (now - self.requests.queue[0])
                if sleep_time > 0:
                    logging.info(f"Rate limiting: waiting {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
            
            self.requests.put(now)

# Global rate limiter
rate_limiter = RateLimiter(max_requests_per_minute=25)  # Conservative limit

def get_subfolder_path(roi_path, keyword):
    """Finds a subfolder in roi_path that contains the given keyword."""
    for item in os.listdir(roi_path):
        item_path = os.path.join(roi_path, item)
        if os.path.isdir(item_path) and keyword in item:
            # More specific check to avoid matching e.g. '..._scl' with '..._cloud_score'
            if keyword.endswith('_scl') and not item.endswith('_scl'):
                continue
            if keyword.endswith('_cloud_score') and not item.endswith('_cloud_score'):
                continue
            return item_path
    return None

def verify_and_clean_roi_data(roi_path):
    """
    Verifies band counts for downloaded GeoTIFFs and cleans incomplete or corrupt files.
    
    Checks for:
    - RVI: 3 bands
    - NDVI Cloud Score: 6 band
    - NDVI SCL: 6 band
    Args:
        roi_path (str): The full path to the ROI folder.

    Returns:
        bool: True if all data is valid, False if any file was deleted.
    """
    is_valid = True
    roi_name = os.path.basename(roi_path)
    
    checks = [
        # {'keyword': '_ndvi8days', 'bands': 6, 'name': 'NDVI'},
        {'keyword': '_rvi_8days', 'bands': 3, 'name': 'RVI'},
        {'keyword': '_ndvi8days_cloud_score', 'bands': 6, 'name': 'NDVI Cloud Score'},
        # {'keyword': '_ndvi8days_scl', 'bands': 1, 'name': 'NDVI SCL'}
    ]

    for check in checks:
        folder_path = get_subfolder_path(roi_path, check['keyword'])
        if not folder_path:
            # If it's a primary folder (NDVI/RVI), its absence means failure.
            if check['name'] in ['NDVI', 'RVI']:
                logging.warning(f"Verification FAILED: {check['name']} folder ('*{check['keyword']}') not found for {roi_name}.")
                is_valid = False
            continue

        logging.info(f"Verifying {check['name']} images in {os.path.basename(folder_path)} for {roi_name}...")
        image_files = glob.glob(os.path.join(folder_path, '*.tif'))
        
        if not image_files and check['name'] in ['NDVI', 'RVI']:
            logging.warning(f"Verification FAILED: No .tif files found in {os.path.basename(folder_path)} for {roi_name}.")
            is_valid = False
            continue

        for tif_file in image_files:
            try:
                with rasterio.open(tif_file) as src:
                    band_count = src.count
                    if band_count != check['bands']:
                        logging.warning(f"DELETING: {os.path.basename(tif_file)} had {band_count} bands, expected {check['bands']}.")
                        is_valid = False
                        src.close()
                        os.remove(tif_file)
            except Exception as e:
                logging.error(f"DELETING corrupt file {os.path.basename(tif_file)}: {e}")
                is_valid = False
                try:
                    os.remove(tif_file)
                except Exception as del_e:
                    logging.error(f"Could not delete corrupt file {tif_file}: {del_e}")
    
    return is_valid

def create_roi_from_tif_metadata(tif_file_path):
    """
    Reads a TIF file's metadata and creates an ROI geometry and resolution information.
    
    Args:
        tif_file_path (str): Path to the TIF file
        
    Returns:
        dict: Dictionary containing:
            - 'geometry': ee.Geometry.Polygon object in EPSG:4326
            - 'resolution': Resolution in meters
            - 'roi_name': ROI name derived from filename
            - 'bounds': Original bounds from the TIF file
            - 'crs': Original CRS of the TIF file
    """
    try:
        with rasterio.open(tif_file_path) as dataset:
            # Get basic metadata
            bounds = dataset.bounds  # (left, bottom, right, top)
            crs = dataset.crs
            transform = dataset.transform
            width = dataset.width
            height = dataset.height
            
            # Calculate resolution in meters
            # If CRS is geographic (degrees), convert to approximate meters
            if crs.is_geographic:
                # Approximate conversion: 1 degree ≈ 111,319.5 meters at equator
                # Use the center latitude for more accurate conversion
                center_lat = (bounds.bottom + bounds.top) / 2
                resolution_x_meters = abs(transform.a) * 111319.5 * math.cos(math.radians(center_lat))
                resolution_y_meters = abs(transform.e) * 111319.5
                resolution = (resolution_x_meters + resolution_y_meters) / 2  # Average resolution
            else:
                # Assuming CRS is in meters (like UTM, Web Mercator, etc.)
                resolution_x_meters = abs(transform.a)
                resolution_y_meters = abs(transform.e)
                resolution = (resolution_x_meters + resolution_y_meters) / 2  # Average resolution
            
            # Transform bounds to EPSG:4326 if needed
            if crs.to_string() != 'EPSG:4326':
                bounds_4326 = transform_bounds(crs, 'EPSG:4326', *bounds)
            else:
                bounds_4326 = bounds
            
            # Create ROI name from filename
            filename = os.path.basename(tif_file_path)
            roi_name = os.path.splitext(filename)[0]
            
            # Create coordinates in counter-clockwise order for ee.Geometry.Polygon
            # bounds_4326 = (left, bottom, right, top)
            coords_4326 = [
                [bounds_4326[0], bounds_4326[1]],  # Bottom-left (left, bottom)
                [bounds_4326[2], bounds_4326[1]],  # Bottom-right (right, bottom)
                [bounds_4326[2], bounds_4326[3]],  # Top-right (right, top)
                [bounds_4326[0], bounds_4326[3]],  # Top-left (left, top)
                [bounds_4326[0], bounds_4326[1]]   # Close the loop
            ]
            
            # Create ee.Geometry.Polygon object
            geometry = ee.Geometry.Polygon(
                coords=coords_4326,
                proj='EPSG:4326',
                geodesic=False,
                evenOdd=True
            )
            
            print(f"TIF Metadata Analysis:")
            print(f"  File: {tif_file_path}")
            print(f"  Original CRS: {crs}")
            print(f"  Dimensions: {width} x {height} pixels")
            print(f"  Resolution: {resolution:.2f} meters")
            print(f"  Bounds (EPSG:4326): {bounds_4326}")
            print(f"  ROI Name: {roi_name}")
            
            return {
                'geometry': geometry,
                'resolution': round(resolution),
                'roi_name': roi_name,
                'bounds': bounds_4326,
                'crs': crs.to_string(),
                'width': width,
                'height': height
            }
            
    except Exception as e:
        print(f"Error reading TIF file {tif_file_path}: {e}")
        return None


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


def read_rois_from_excel(excel_file_path, kv_type_prefix):
    """
    Reads an Excel file to extract ROI information and returns a dictionary
    of ROI names to ee.Geometry.Polygon objects.

    The Excel file must contain columns: 'Box_ID', 'Longitude', 'Latitude',
    'Width_degrees', 'Height_degrees'.

    Args:
        excel_file_path (str): Path to the Excel file.
        kv_type_prefix (str): Prefix for ROI name (e.g., "220kV" or "500kV").

    Returns:
        dict: Dictionary where keys are ROI names (e.g., "220kV_Box1")
              and values are ee.Geometry.Polygon objects in EPSG:4326, planar.
    """
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading Excel file {excel_file_path}: {e}")
        return {}

    roi_geometries = {}
    required_columns = ['Box_ID', 'Longitude', 'Latitude', 'Width_degrees', 'Height_degrees']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Excel file {excel_file_path} is missing one or more required columns: {required_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return {}

    for index, row in df.iterrows():
        try:
            box_id = row['Box_ID']
            center_lon = float(row['Longitude'])
            center_lat = float(row['Latitude'])
            width_deg = float(row['Width_degrees'])
            height_deg = float(row['Height_degrees'])

            min_lon = center_lon - (width_deg / 2)
            max_lon = center_lon + (width_deg / 2)
            min_lat = center_lat - (height_deg / 2)
            max_lat = center_lat + (height_deg / 2)

            roi_name = f"{kv_type_prefix}_{box_id}"
            
            # Create an ee.Geometry.Polygon object
            # Coordinates are in EPSG:4326 (lon/lat degrees)
            # Using Counter-Clockwise (CCW) order: BL, BR, TR, TL
            coords_4326 = [
                [min_lon, min_lat],  # Bottom-left
                [max_lon, min_lat],  # Bottom-right
                [max_lon, max_lat],  # Top-right
                [min_lon, max_lat],  # Top-left
                [min_lon, min_lat]   # Close the loop
            ]
            
            geometry = ee.Geometry.Polygon(
                coords=coords_4326,
                proj='EPSG:4326',
                geodesic=False,  # Treat as planar in EPSG:4326 for consistency
                evenOdd=True    # Use even-odd rule for interior determination
            )
            roi_geometries[roi_name] = geometry
        except ValueError as ve:
            print(f"Skipping row {index+2} in {excel_file_path} (Box_ID: {row.get('Box_ID', 'N/A')}) due to data type error: {ve}. Ensure numeric values for coordinates/dimensions.")
        except Exception as e:
            print(f"Error processing row {index+2} in {excel_file_path} for Box_ID {row.get('Box_ID', 'N/A')}: {e}")
            
    return roi_geometries

def create_rois_from_coordinates_dict(coordinates_dict, resolutions=[10, 20], pixels=695):
    """
    Creates ROI geometries from a dictionary of coordinates for specified resolutions and pixel sizes.
    
    Args:
        coordinates_dict (dict): Dictionary with ROI names as keys and (longitude, latitude) tuples as values
        resolutions (list): List of resolutions in meters (default: [10, 20])
        pixels (int): Number of pixels per side for square images (default: 512)
    
    Returns:
        dict: Dictionary where keys are ROI names with resolution suffix (e.g., "ROI1_10m", "ROI1_20m")
              and values are ee.Geometry.Polygon objects
    """
    roi_geometries = {}
    
    for roi_name, (center_lon, center_lat) in coordinates_dict.items():
        try:
            center_lon = float(center_lon)
            center_lat = float(center_lat)
            
            # Validate coordinates
            if not (-180 <= center_lon <= 180):
                print(f"Warning: Invalid longitude {center_lon} for ROI {roi_name}. Skipping.")
                continue
            if not (-90 <= center_lat <= 90):
                print(f"Warning: Invalid latitude {center_lat} for ROI {roi_name}. Skipping.")
                continue
            
            for resolution in resolutions:
                # Calculate the size of the bounding box in meters
                box_size_meters = pixels * resolution
                
                # Convert meters to degrees
                # 1 degree latitude ≈ 111,319.5 meters (constant)
                # 1 degree longitude ≈ 111,319.5 * cos(latitude) meters
                lat_deg_per_meter = 1 / 111319.5
                lon_deg_per_meter = 1 / (111319.5 * math.cos(math.radians(center_lat)))
                
                # Calculate half the box size in degrees
                half_box_lat = (box_size_meters / 2) * lat_deg_per_meter
                half_box_lon = (box_size_meters / 2) * lon_deg_per_meter
                
                # Calculate bounding box coordinates
                min_lon = center_lon - half_box_lon
                max_lon = center_lon + half_box_lon
                min_lat = center_lat - half_box_lat
                max_lat = center_lat + half_box_lat
                
                # Create ROI name with resolution suffix
                roi_name_with_res = f"{roi_name}_{resolution}m"
                
                # Create coordinates in counter-clockwise order for ee.Geometry.Polygon
                coords_4326 = [
                    [min_lon, min_lat],  # Bottom-left
                    [max_lon, min_lat],  # Bottom-right
                    [max_lon, max_lat],  # Top-right
                    [min_lon, max_lat],  # Top-left
                    [min_lon, min_lat]   # Close the loop
                ]
                
                # Create ee.Geometry.Polygon object
                geometry = ee.Geometry.Polygon(
                    coords=coords_4326,
                    proj='EPSG:4326',
                    geodesic=False,
                    evenOdd=True
                )
                
                roi_geometries[roi_name_with_res] = geometry
                
                print(f"Created ROI: {roi_name_with_res} - Center: ({center_lon:.6f}, {center_lat:.6f}), "
                      f"Size: {box_size_meters}m x {box_size_meters}m ({pixels}x{pixels} pixels at {resolution}m resolution)")
                
        except ValueError as ve:
            print(f"Error processing ROI {roi_name}: Invalid coordinate values. {ve}")
        except Exception as e:
            print(f"Error processing ROI {roi_name}: {e}")
    
    return roi_geometries


def read_coordinates_from_excel(excel_file_path):
    """
    Reads an Excel file to extract ROI information and returns a dictionary
    of ROI names to (longitude, latitude) tuples.

    The Excel file must contain columns: 'code', 'PhienHieu', 'long', 'lat'.

    Args:
        excel_file_path (str): Path to the Excel file.

    Returns:
        dict: Dictionary where keys are ROI names (e.g., "code_PhienHieu")
              and values are (longitude, latitude) tuples.
    """
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading Excel file {excel_file_path}: {e}")
        return {}

    coordinates_dict = {}
    required_columns = ['code', 'PhienHieu', 'lon', 'lat']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Excel file {excel_file_path} is missing one or more required columns: {required_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return {}

    for index, row in df.iterrows():
        try:
            code = row['code']
            phien_hieu = row['PhienHieu']
            center_lon = float(row['lon'])
            center_lat = float(row['lat'])

            roi_name = f"{code}_{phien_hieu}"
            
            coordinates_dict[roi_name] = (center_lon, center_lat)

        except (ValueError, TypeError) as ve:
            print(f"Skipping row {index+2} in {excel_file_path} due to data type error: {ve}. Ensure numeric values for coordinates.")
        except Exception as e:
            print(f"Error processing row {index+2} in {excel_file_path}: {e}")
            
    return coordinates_dict


def process_single_roi(roi_item, start_date_ee, end_date_ee, big_folder, 
                       crawl_rvi=True, crawl_ndvi=True, 
                       enable_parallel_downloads=True, generate_date_mapping=False, max_retries=5):
    """
    Process a single ROI with rate limiting, error handling, and verification/recrawl logic.
    
    Args:
        roi_item: Tuple of (roi_name, geometry_obj)
        start_date_ee: Start date for data retrieval
        end_date_ee: End date for data retrieval
        big_folder: Output directory
        crawl_rvi (bool): Whether to crawl RVI data.
        crawl_ndvi (bool): Whether to crawl NDVI data.
        enable_parallel_downloads: Enable parallel processing for individual band downloads
        generate_date_mapping: Enable RVI date mapping generation (may cause hanging)
        max_retries: Maximum number of processing attempts for the ROI.
    
    Returns:
        dict: Processing result with status and any errors
    """
    roi_name, geometry_obj = roi_item
    
    for attempt in range(max_retries):
        result = {
            'roi_name': roi_name,
            'status': 'failed',
            'error': None,
            'start_time': time.time(),
            'attempt': attempt + 1
        }
    
        try:
            logging.info(f"--- START {roi_name} (Attempt {attempt + 1}/{max_retries}) ---")
            
            # Apply rate limiting before making GEE requests
            rate_limiter.wait_if_needed()
            
            if crawl_rvi:
                logging.info(f"Retrieving RVI for {roi_name}...")
                # Retrieve RVI (expects order: start, end, geometry, big_folder, roi_name, enable_parallel, generate_date_mapping)
                main_rvi(start_date_ee, end_date_ee, geometry_obj, big_folder, roi_name, 
                         enable_parallel=enable_parallel_downloads, generate_date_mapping=generate_date_mapping)
                
                # Small delay between RVI and NDVI to avoid overwhelming GEE
                time.sleep(2)
                rate_limiter.wait_if_needed()

            if crawl_ndvi:
                logging.info(f"Retrieving NDVI for {roi_name}...")
                # Retrieve NDVI (expects order: start, end, geometry, roi_name, big_folder, enable_parallel)
                main_ndvi(start_date_ee, end_date_ee, geometry_obj, roi_name, big_folder, enable_parallel=enable_parallel_downloads)

            # --- VERIFICATION STEP ---
            roi_path = os.path.join(big_folder, roi_name)
            if os.path.exists(roi_path):
                is_data_valid = verify_and_clean_roi_data(roi_path)
                if is_data_valid:
                    logging.info(f"--- VERIFIED {roi_name} ---")
                    result['status'] = 'success'
                    break # Success, exit retry loop
                else:
                    logging.warning(f"--- VERIFICATION FAILED for {roi_name}. Retrying... ---")
                    result['error'] = 'Verification failed, data was incomplete.'
                    # Loop will continue to next attempt if retries are left
            else:
                logging.error(f"ROI path {roi_path} not found after download attempt.")
                result['error'] = 'ROI folder not created.'
                # Loop will continue to next attempt if retries are left

        except ee.EEException as gee_err:
            error_msg = f"[GEE ERROR] {roi_name}: {gee_err}"
            logging.error(error_msg)
            result['error'] = str(gee_err)
            
        except Exception as ex:
            error_msg = f"[ERROR] {roi_name}: {ex}"
            logging.error(error_msg)
            result['error'] = str(ex)

        # If it's not the last attempt, wait before retrying
        if attempt < max_retries - 1:
            logging.info(f"Waiting 5s before retry for {roi_name}...")
            time.sleep(5)

    # Final result logging after all attempts
    result['end_time'] = time.time()
    result['duration'] = result.get('end_time', result['start_time']) - result['start_time']

    if result['status'] == 'success':
        logging.info(f"--- COMPLETED {roi_name} in {result['duration']:.1f}s after {result['attempt']} attempt(s) ---")
    else:
        logging.error(f"--- FAILED {roi_name} after {max_retries} attempts. Last error: {result['error']} ---")
    
    return result


def process_rois_parallel(rois, start_date_ee, end_date_ee, big_folder, 
                        max_workers=3, crawl_rvi=True, crawl_ndvi=True, 
                        enable_parallel_downloads=True, generate_date_mapping=False):
    """
    Process ROIs in parallel with controlled concurrency.
    
    Args:
        rois: Dictionary of ROI names to geometry objects
        start_date_ee: Start date for data retrieval
        end_date_ee: End date for data retrieval
        big_folder: Output directory
        max_workers: Maximum number of concurrent workers (default: 3)
        crawl_rvi (bool): Whether to crawl RVI data.
        crawl_ndvi (bool): Whether to crawl NDVI data.
        enable_parallel_downloads: Enable parallel processing for individual band downloads
        generate_date_mapping: Enable RVI date mapping generation (may cause hanging)
    
    Returns:
        list: List of processing results
    """
    roi_items = list(rois.items())
    total_rois = len(roi_items)
    results = []
    
    logging.info(f"Starting parallel processing of {total_rois} ROIs with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_roi = {
            executor.submit(process_single_roi, roi_item, start_date_ee, end_date_ee, big_folder,
                            crawl_rvi, crawl_ndvi,
                            enable_parallel_downloads, generate_date_mapping): roi_item[0]
            for roi_item in roi_items
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_roi):
            roi_name = future_to_roi[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                status_emoji = "✅" if result['status'] == 'success' else "❌"
                duration = result.get('duration', 0)
                logging.info(f"{status_emoji} Progress: {completed}/{total_rois} - {roi_name} ({duration:.1f}s)")
                
            except Exception as exc:
                error_result = {
                    'roi_name': roi_name,
                    'status': 'failed',
                    'error': str(exc),
                    'start_time': time.time(),
                    'end_time': time.time()
                }
                results.append(error_result)
                completed += 1
                logging.error(f"❌ Progress: {completed}/{total_rois} - {roi_name} failed with exception: {exc}")
    
    return results


def print_processing_summary(results):
    """Print a summary of processing results."""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    
    total_duration = sum(r.get('duration', 0) for r in results if 'duration' in r)
    avg_duration = total_duration / successful if successful > 0 else 0
    
    logging.info("\n" + "="*60)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*60)
    logging.info(f"Total ROIs processed: {total}")
    logging.info(f"Successful: {successful} ✅")
    logging.info(f"Failed: {failed} ❌")
    logging.info(f"Success rate: {(successful/total)*100:.1f}%")
    logging.info(f"Average processing time: {avg_duration:.1f}s per ROI")
    logging.info(f"Total processing time: {total_duration:.1f}s")
    
    if failed > 0:
        logging.info("\nFailed ROIs:")
        for result in results:
            if result['status'] == 'failed':
                logging.info(f"  - {result['roi_name']}: {result.get('error', 'Unknown error')}")
    
    logging.info("="*60)


if __name__ == "__main__":
    """ENTRY POINT – Crawl NDVI & RVI for predefined center coordinates
    ------------------------------------------------------------------
    This version no longer scans ROI folders.  Instead it builds 512 × 512 pixel
    (10-m resolution) bounding-boxes around a hard-coded set of centre
    coordinates, then retrieves NDVI and RVI time-series for each box.
    """

    # ---------------------------------------------------------------------
    # HYPERPARAMETERS & CONFIGURATION
    # ---------------------------------------------------------------------
    
    # --- Data sources ---
    excel_file_path = '/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_retrieval/forest_grids_balanced.xlsx'
    big_folder = "/mnt/hdd12tb/code/nhatvm/BRIOS/BRIOS/data_Tung_ndvi_s2_combined"
    
    # --- Time range ---
    date_start_str = '2019-01-01'
    date_end_str   = '2025-01-01'

    # --- Crawling Control ---
    # Set to True to crawl RVI, False to skip
    CRAWL_RVI = True
    # Set to True to crawl NDVI, False to skip
    CRAWL_NDVI = True

    # --- ROI Partitioning ---
    # To process all ROIs, set ROI_SLICE_START = 0 and ROI_SLICE_END = None.
    # To process a subset, define the start and end index. For example, to
    # process ROIs from index 100 up to (but not including) 200, set:
    # ROI_SLICE_START = 100
    # ROI_SLICE_END = 200
    # This is useful for splitting a large job into smaller batches.
    ROI_SLICE_START = 200      # Inclusive index, starts from 0
    ROI_SLICE_END = 250     # Exclusive index (e.g., 10 processes up to index 9)

    # --- Performance ---
    MAX_WORKERS = 4
    ENABLE_PARALLEL = True
    ENABLE_PARALLEL_DOWNLOADS = True
    GENERATE_DATE_MAPPING = True # Can be slow

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    start_date_ee  = ee.Date(date_start_str)
    end_date_ee    = ee.Date(date_end_str)
    
    coordinates_dict = read_coordinates_from_excel(excel_file_path)

    if not coordinates_dict:
        logging.error("No coordinates were loaded from the Excel file. Exiting.")
        exit(1)

    # ------------------------------------------------------------------
    # Build ROIs and apply partitioning
    # ------------------------------------------------------------------
    all_rois = create_rois_from_coordinates_dict(coordinates_dict, resolutions=[10], pixels=695)
    
    # Apply slicing to the ROIs
    roi_items = list(all_rois.items())
    if ROI_SLICE_END is None:
        ROI_SLICE_END = len(roi_items)
    
    sliced_roi_items = roi_items[ROI_SLICE_START:ROI_SLICE_END]
    rois = dict(sliced_roi_items)

    logging.info(f"Prepared {len(all_rois)} total ROI geometries, processing slice [{ROI_SLICE_START}:{ROI_SLICE_END}] -> {len(rois)} ROIs")
    logging.info(f"Processing configuration: MAX_WORKERS={MAX_WORKERS}, PARALLEL={ENABLE_PARALLEL}, PARALLEL_DOWNLOADS={ENABLE_PARALLEL_DOWNLOADS}")
    logging.info(f"Crawling: RVI={'ON' if CRAWL_RVI else 'OFF'}, NDVI={'ON' if CRAWL_NDVI else 'OFF'}")
    logging.info(f"Date range: {date_start_str} to {date_end_str}")
    logging.info(f"Output folder: {big_folder}")

    # ------------------------------------------------------------------
    # Process ROIs (parallel or sequential)
    # ------------------------------------------------------------------
    start_time = time.time()
    
    if ENABLE_PARALLEL and len(rois) > 1:
        # Parallel processing
        results = process_rois_parallel(rois, start_date_ee, end_date_ee, big_folder, 
                                        max_workers=MAX_WORKERS,
                                        crawl_rvi=CRAWL_RVI, crawl_ndvi=CRAWL_NDVI,
                                        enable_parallel_downloads=ENABLE_PARALLEL_DOWNLOADS, 
                                        generate_date_mapping=GENERATE_DATE_MAPPING)
        print_processing_summary(results)
    else:
        # Sequential processing (fallback or single ROI)
        logging.info("Using sequential processing")
        results = []
        
        for roi_name, geometry_obj in rois.items():
            roi_item = (roi_name, geometry_obj)
            result = process_single_roi(roi_item, start_date_ee, end_date_ee, big_folder,
                                        crawl_rvi=CRAWL_RVI, crawl_ndvi=CRAWL_NDVI,
                                        enable_parallel_downloads=ENABLE_PARALLEL_DOWNLOADS, 
                                        generate_date_mapping=GENERATE_DATE_MAPPING)
            results.append(result)
        
        print_processing_summary(results)

    total_time = time.time() - start_time
    logging.info(f"\nAll ROI processing complete in {total_time:.1f} seconds.")
    
    # Save results to file for later analysis
    results_file = os.path.join(big_folder, f"processing_results_{int(time.time())}.json")
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Processing results saved to: {results_file}")
    except Exception as e:
        logging.warning(f"Could not save results file: {e}")



