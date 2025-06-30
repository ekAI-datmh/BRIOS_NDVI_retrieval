import ee
import os
import requests
import tempfile
import zipfile
import shutil
import time
import json
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('ndvi_retrieval.log')  # File output
    ]
)

# Thread-safe rate limiter for downloads
download_lock = threading.Lock()
last_download_time = 0
MIN_DOWNLOAD_INTERVAL = 1.0  # Minimum seconds between downloads

def rate_limited_download():
    """Ensure minimum interval between downloads"""
    global last_download_time
    with download_lock:
        current_time = time.time()
        time_since_last = current_time - last_download_time
        if time_since_last < MIN_DOWNLOAD_INTERVAL:
            sleep_time = MIN_DOWNLOAD_INTERVAL - time_since_last
            time.sleep(sleep_time)
        last_download_time = time.time()

# =============================================================================
# FUNCTIONS
# =============================================================================
def coor_to_geometry(json_file: str):
    """Loads coordinates from a GeoJSON file and converts to ee.Geometry.Polygon."""
    try:
        with open(json_file, 'r') as f:
            geojson = json.load(f)
            # Handle different GeoJSON types if necessary, assuming Polygon for now
            if geojson['type'] == 'FeatureCollection':
                coor_list = geojson['features'][0]['geometry']['coordinates']
            elif geojson['type'] == 'Feature':
                coor_list = geojson['geometry']['coordinates']
            elif geojson['type'] == 'Polygon':
                coor_list = geojson['coordinates']
            else:
                raise ValueError(f"Unsupported GeoJSON type: {geojson['type']}")
        logging.info(f"Successfully loaded ROI geometry from {json_file}")
        return ee.Geometry.Polygon(coor_list)
    except FileNotFoundError:
        logging.critical(f"ROI JSON file not found: {json_file}")
        raise
    except json.JSONDecodeError as e:
        logging.critical(f"Error decoding JSON from {json_file}: {e}")
        raise
    except Exception as e:
        logging.critical(f"An unexpected error occurred while processing ROI geometry from {json_file}: {e}")
        raise

def get_sentinel_collection_cloud_score_plus(start_date, end_date, roi):
    """
    Loads the Sentinel-2 collection, applies initial filters and cloud masking using Cloud Score+.
    """
    logging.info(f"Fetching Sentinel-2 collection with Cloud Score+ masking for dates {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
    
    # Advance dates by ±8 days to ensure full coverage for 8-day composites
    s_date = start_date.advance(-8, 'day')
    e_date = end_date.advance(8, 'day')
    
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    qa_band = 'cs'
    clear_threshold = 0.5 # Pixels with cloud score >= 0.5 are considered cloudy
    
    sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(s_date, e_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85)) # Initial broad cloud filter
                 .linkCollection(cs_plus, [qa_band]))

    # Apply cloud mask using Cloud Score+ (masking out cloudy pixels)
    sentinel_masked_cloud = sentinel2.map(
        lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold)).clip(roi)
    )
    logging.info(f"Applied Cloud Score+ masking to Sentinel-2 collection")
    return sentinel_masked_cloud

def get_sentinel_collection_scl_masking(start_date, end_date, roi):
    """
    Loads the Sentinel-2 collection, applies initial filters and cloud masking using SCL band.
    """
    logging.info(f"Fetching Sentinel-2 collection with SCL masking for dates {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
    
    # Advance dates by ±8 days to ensure full coverage for 8-day composites
    s_date = start_date.advance(-8, 'day')
    e_date = end_date.advance(8, 'day')
    
    sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(s_date, e_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85))) # Initial broad cloud filter

    # Apply cloud mask using SCL band
    # SCL values to mask out: 0 (No Data), 1 (Saturated), 3 (Cloud Shadows), 
    # 8 (Cloud Medium Probability), 9 (Cloud High Probability), 10 (Thin Cirrus), 11 (Snow)
    # Keep: 2 (Dark Area Pixels), 4 (Vegetation), 5 (Not Vegetated), 6 (Water), 7 (Unclassified)
    def mask_scl_clouds(image):
        scl = image.select('SCL')
        # Create mask for clear pixels (values 2, 4, 5, 6, 7)
        clear_pixels = scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return image.updateMask(clear_pixels).clip(roi)
    
    sentinel_masked_scl = sentinel2.map(mask_scl_clouds)
    logging.info(f"Applied SCL masking to Sentinel-2 collection")
    return sentinel_masked_scl


def separate_collections(ndvi_collection):
    """
    Separates a collection into cloud-free and cloudy subsets based on the 
    'CLOUDY_PIXEL_PERCENTAGE' property.
    """
    cloud_free = ndvi_collection.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 2)) # Very low cloud percentage
    cloudy = ndvi_collection.filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE', 2)) # More than 2% cloudy
    
    cloudy_count = cloudy.size().getInfo()
    cloud_free_count = cloud_free.size().getInfo()
    
    logging.info(f'Cloudy images count (CLOUDY_PIXEL_PERCENTAGE > 2%): {cloudy_count}')
    # logging.info(f'Cloud-free images count (CLOUDY_PIXEL_PERCENTAGE <= 2%): {cloud_free_count}')
    
    return {'cloudFree': cloud_free, 'cloudy': cloudy}

def calculate_8day_composites_with_bands(image_collection, start_date, end_date, exclude_date, bands=['B2', 'B3', 'B4', 'B8', 'SCL']):
    """
    For each 8-day period, creates a composite image with specified bands and calculated NDVI.
    If no images exist in the period, returns an empty image with a placeholder time.
    """
    days_step = 8
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    millis_step = days_step * 24 * 60 * 60 * 1000
    list_of_dates = ee.List.sequence(start.millis(), end.millis(), millis_step)
    
    logging.info(f"Calculating 8-day composites with bands {bands} and NDVI from {start.format('YYYY-MM-dd').getInfo()} to {end.format('YYYY-MM-dd').getInfo()}")

    def composite_for_millis(millis):
        composite_center = ee.Date(millis)
        # Define the 8-day window centered around composite_center
        composite_start = composite_center.advance(- (days_step / 2), 'day')
        composite_end = composite_center.advance((days_step / 2), 'day')
        period_collection = image_collection.filterDate(composite_start, composite_end)
        
        composite_image = ee.Algorithms.If(
            period_collection.size().gt(0),
            (period_collection.median()
             .select(bands)
             .addBands(period_collection.median().normalizedDifference(['B8', 'B4']).rename('NDVI'))
             .unmask(-100) # Unmask with a NoData value of -100
             .set('system:time_start', composite_center.millis())),
            ee.Image().set('system:time_start', exclude_date) # Placeholder for empty periods
        )
        return composite_image
    
    composites = ee.ImageCollection(list_of_dates.map(composite_for_millis))
    logging.info(f"Generated {composites.size().getInfo()} 8-day composites with bands and NDVI (including placeholders).")
    return composites

def download_single_band(image, band_name, date_str, roi, temp_dir, max_retries=3):
    """
    Downloads a single band from an image and returns the path to the downloaded TIFF file.
    """
    rate_limited_download()  # Apply rate limiting
    
    params = {
        'scale': 10,
        'region': roi,
        'fileFormat': 'ZIP',
        'maxPixels': 1e13
    }
    
    download_url = image.select(band_name).getDownloadURL(params)
    
    for attempt in range(max_retries):
        zip_path = os.path.join(temp_dir, f"{band_name}_{date_str}.zip")
        
        try:
            logging.debug(f"Downloading band {band_name} for {date_str} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(download_url, timeout=300)
            response.raise_for_status()

            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Unzip the file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    tif_in_zip = [f for f in zip_ref.namelist() if f.lower().endswith('.tif')]
                    if tif_in_zip:
                        extracted_tif_name = tif_in_zip[0]
                        zip_ref.extract(extracted_tif_name, temp_dir)
                        extracted_tif_path = os.path.join(temp_dir, extracted_tif_name)
                        
                        # Rename to band-specific name
                        band_tif_path = os.path.join(temp_dir, f"{band_name}_{date_str}.tif")
                        shutil.move(extracted_tif_path, band_tif_path)
                        
                        # Clean up zip file
                        os.remove(zip_path)
                        
                        logging.debug(f"Successfully downloaded band {band_name} for {date_str}")
                        return band_tif_path
                    else:
                        logging.warning(f"No TIFF file found in ZIP for band {band_name}, {date_str}")
            else:
                logging.warning(f"Download failed for band {band_name}, {date_str} (Status code: {response.status_code})")

        except requests.exceptions.RequestException as e:
            logging.warning(f"Request error for band {band_name}, {date_str} (attempt {attempt+1}): {e}")
        except zipfile.BadZipFile:
            logging.warning(f"Downloaded file is a bad ZIP for band {band_name}, {date_str}")
        except Exception as e:
            logging.error(f"Unhandled error downloading band {band_name}, {date_str} (attempt {attempt+1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))
    
    logging.error(f"Failed to download band {band_name} for {date_str} after {max_retries} attempts")
    return None

def download_bands_parallel_ndvi(image, roi, date_str, temp_dir, bands=['B2', 'B3', 'B4', 'B8', 'SCL', 'NDVI'], max_workers=3):
    """
    Download multiple bands in parallel for a single NDVI image.
    
    Args:
        image: Earth Engine image
        roi: Region of interest
        date_str: Date string for naming
        temp_dir: Temporary directory for downloads
        bands: List of band names to download
        max_workers: Maximum number of concurrent downloads (default: 3)
    
    Returns:
        dict: Dictionary mapping band names to downloaded file paths (None if failed)
    """
    results = {}
    
    # Check which bands are available in the image
    image_bands = image.bandNames().getInfo()
    available_bands = [band for band in bands if band in image_bands]
    
    if not available_bands:
        logging.warning(f"No required bands found in image for {date_str}")
        return results
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_band = {
            executor.submit(download_single_band, image, band, date_str, roi, temp_dir): band
            for band in available_bands
        }
        
        # Collect results
        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                result = future.result()
                results[band] = result
                if result:
                    logging.debug(f"Successfully downloaded {band} for {date_str}")
                else:
                    logging.warning(f"Failed to download {band} for {date_str}")
            except Exception as exc:
                logging.error(f"Exception downloading {band} for {date_str}: {exc}")
                results[band] = None
    
    return results

def merge_bands_to_multiband(band_files, output_path, band_names):
    """
    Merges individual band TIFF files into a single multi-band TIFF file.
    """
    try:
        # Read the first band to get metadata
        with rasterio.open(band_files[0]) as src:
            profile = src.profile.copy()
            height, width = src.shape
            transform = src.transform
            crs = src.crs
        
        # Update profile for multi-band output
        profile.update({
            'count': len(band_files),
            'dtype': 'float32'
        })
        
        # Create the multi-band file
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, (band_file, band_name) in enumerate(zip(band_files, band_names), 1):
                if band_file and os.path.exists(band_file):
                    with rasterio.open(band_file) as src:
                        data = src.read(1).astype('float32')
                        dst.write(data, i)
                        dst.set_band_description(i, band_name)
                else:
                    # Write nodata for missing bands
                    nodata_array = np.full((height, width), -100, dtype='float32')
                    dst.write(nodata_array, i)
                    dst.set_band_description(i, band_name)
        
        logging.info(f"Successfully created multi-band file: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating multi-band file {output_path}: {e}")
        return False

def download_multiband_composites(composites, big_folder, roi, roi_name, folder_suffix, bands=['B2', 'B3', 'B4', 'B8', 'SCL', 'NDVI'], enable_parallel=True):
    """
    Downloads each composite as individual bands, then merges them into multi-band TIFF files.
    """
    image_list = composites.toList(composites.size())
    size = composites.size().getInfo()
    out_folder = os.path.join(big_folder, roi_name, f'{roi_name}_ndvi8days_{folder_suffix}')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        logging.info(f"Created output folder for multi-band downloads: {out_folder}")

    logging.info(f"Starting download of {size} multi-band composites for ROI '{roi_name}' with {folder_suffix} masking (parallel={enable_parallel}).")

    successful_downloads = 0
    failed_downloads = 0

    for i in range(size):
        image = ee.Image(image_list.get(i))
        
        # Check if this is not a placeholder image
        exclude_date_value = ee.Date('1900-01-01').millis().getInfo()
        if image.get('system:time_start').getInfo() == exclude_date_value:
            logging.debug(f"Skipping empty or placeholder image at index {i}.")
            continue

        # Check if image has any of the required bands
        image_bands = image.bandNames().getInfo()
        available_bands = [band for band in bands if band in image_bands]
        if not available_bands:
            logging.debug(f"Skipping image at index {i}, no required bands found.")
            continue

        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        # Construct the expected final file path
        final_tif_path = os.path.join(out_folder, f"ndvi8days_{date_str}.tif")

        # Check if the file already exists
        if os.path.exists(final_tif_path):
            logging.info(f"Skipping download for {date_str} ({folder_suffix}), file already exists: {final_tif_path}")
            continue

        # Create temporary directory for this date
        temp_dir = tempfile.mkdtemp()
        
        try:
            logging.info(f"Downloading bands for {date_str} with {folder_suffix} masking...")
            
            if enable_parallel:
                # Use parallel downloading
                download_results = download_bands_parallel_ndvi(image, roi, date_str, temp_dir, bands)
                
                # Convert results to the expected format
                band_files = []
                successful_bands = []
                
                for band in bands:
                    if band in download_results and download_results[band]:
                        band_files.append(download_results[band])
                        successful_bands.append(band)
                    else:
                        band_files.append(None)
                        successful_bands.append(band)
                        if band in image_bands:
                            logging.warning(f"Band {band} download failed for {date_str}")
                        
            else:
                # Sequential downloading (original approach)
                band_files = []
                successful_bands = []
                
                for band in bands:
                    if band in image_bands:
                        band_file = download_single_band(image, band, date_str, roi, temp_dir)
                        band_files.append(band_file)
                        successful_bands.append(band)
                        
                        # Add delay between band downloads to avoid quota issues
                        time.sleep(1)
                    else:
                        band_files.append(None)
                        successful_bands.append(band)
                        logging.warning(f"Band {band} not available for {date_str}")
            
            # Merge bands into multi-band file
            if any(band_files):
                success = merge_bands_to_multiband(band_files, final_tif_path, successful_bands)
                if success:
                    logging.info(f"Successfully created multi-band file for {date_str} with {folder_suffix} masking")
                    successful_downloads += 1
                else:
                    logging.error(f"Failed to create multi-band file for {date_str} with {folder_suffix} masking")
                    failed_downloads += 1
            else:
                logging.error(f"No bands were successfully downloaded for {date_str} with {folder_suffix} masking")
                failed_downloads += 1
        
        except Exception as e:
            logging.error(f"Error processing date {date_str} with {folder_suffix} masking: {e}")
            failed_downloads += 1
        
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logging.debug(f"Cleaned up temporary folder: {temp_dir}")

    logging.info(f"Download completed for {roi_name} ({folder_suffix}): {successful_downloads} successful, {failed_downloads} failed")

def process_cloud_masking_method(masking_method, start_date_ee, end_date_ee, roi, roi_name, big_folder, exclude_date, bands_to_process, enable_parallel=True):
    """
    Process a single cloud masking method and download the results.
    """
    logging.info(f"=== Processing {masking_method} cloud masking method ===")
    
    # Get Sentinel-2 collection with the specified masking method
    if masking_method == "cloud_score_plus":
        sentinel_collection = get_sentinel_collection_cloud_score_plus(start_date_ee, end_date_ee, roi)
        folder_suffix = "cloud_score"
    elif masking_method == "scl":
        sentinel_collection = get_sentinel_collection_scl_masking(start_date_ee, end_date_ee, roi)
        folder_suffix = "scl"
    else:
        logging.error(f"Unknown masking method: {masking_method}")
        return
    
    # Separate the collection into cloud-free and cloudy subsets
    collections = separate_collections(sentinel_collection)
    
    # Create multi-band composites from the cloudy collection
    multiband_composites = calculate_8day_composites_with_bands(collections['cloudy'], start_date_ee, end_date_ee, exclude_date, bands_to_process)

    # Filter out empty placeholders before attempting download
    valid_composites = multiband_composites.filter(ee.Filter.neq('system:time_start', exclude_date))
    valid_composites_count = valid_composites.size().getInfo()
    logging.info(f'Found {valid_composites_count} valid 8-day multi-band composites for download with {masking_method} masking (after removing empty placeholders).')

    if valid_composites_count == 0:
        logging.warning(f"No valid composites found for ROI '{roi_name}' with {masking_method} masking in the period. Skipping download.")
        return

    # Download multi-band composites to local storage
    all_bands = bands_to_process + ['NDVI']
    download_multiband_composites(valid_composites, big_folder, roi, roi_name, folder_suffix, all_bands, enable_parallel=enable_parallel)
    
    logging.info(f"=== Finished processing {masking_method} cloud masking method ===")

def main_ndvi(start_date, end_date, roi, roi_name, big_folder, enable_parallel=True):
    """
    Main function to orchestrate multi-band Sentinel-2 data retrieval with multiple cloud masking methods.
    """

    logging.info(f"--- Starting Sentinel-2 multi-band retrieval process for ROI '{roi_name}' from {start_date} to {end_date} (parallel={enable_parallel}) ---")
    
    # Define time period and region of interest.
    exclude_date = ee.Date('1900-01-01').millis() # Placeholder for empty composites
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)
    bands_to_process = ['B2', 'B3', 'B4', 'B8', 'SCL']

    # Process both cloud masking methods
    cloud_masking_methods = ["cloud_score_plus"]
    
    for masking_method in cloud_masking_methods:
        try:
            process_cloud_masking_method(masking_method, start_date_ee, end_date_ee, roi, roi_name, big_folder, exclude_date, bands_to_process, enable_parallel=enable_parallel)
        except Exception as e:
            logging.error(f"Failed to process {masking_method} masking method: {e}")
            continue

    logging.info(f"--- Sentinel-2 multi-band retrieval process finished for ROI '{roi_name}'. ---")


