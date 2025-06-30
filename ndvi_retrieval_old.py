import ee
import os
import requests
import tempfile
import zipfile
import shutil
import time
import json
# import logging # Import the logging module
from dotenv import load_dotenv
from logger.Logger import app_logger as logging
load_dotenv()


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

def get_sentinel_collection(start_date, end_date, roi):
    """
    Loads the Sentinel-2 collection, applies initial filters and cloud masking.
    """
    logging.info(f"Fetching Sentinel-2 collection for dates {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
    
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
        lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold).Not()).clip(roi)
    )
    # logging.info(f"Initial Sentinel-2 collection size (before cloud masking): {sentinel2.size().getInfo()}")
    # logging.info(f"Sentinel-2 collection size after cloud masking: {sentinel_masked_cloud.size().getInfo()}")
    return sentinel_masked_cloud


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

def calculate_8day_composites(image_collection, start_date, end_date, exclude_date):
    """
    For each 8-day period, creates a composite NDVI image. If no images exist 
    in the period, returns an empty image with a placeholder time.
    """
    days_step = 8
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    millis_step = days_step * 24 * 60 * 60 * 1000
    list_of_dates = ee.List.sequence(start.millis(), end.millis(), millis_step)
    
    logging.info(f"Calculating 8-day NDVI composites from {start.format('YYYY-MM-dd').getInfo()} to {end.format('YYYY-MM-dd').getInfo()}")

    def composite_for_millis(millis):
        composite_center = ee.Date(millis)
        # Define the 8-day window centered around composite_center
        composite_start = composite_center.advance(- (days_step / 2), 'day')
        composite_end = composite_center.advance((days_step / 2), 'day')
        period_collection = image_collection.filterDate(composite_start, composite_end)
        
        composite_image = ee.Algorithms.If(
            period_collection.size().gt(0),
            (period_collection.median()
             .normalizedDifference(['B8', 'B4']) # B8 is NIR, B4 is Red for Sentinel-2
             .rename('NDVI')
             .unmask(-100) # Unmask with a NoData value of -100
             .set('system:time_start', composite_center.millis())),
            ee.Image().set('system:time_start', exclude_date) # Placeholder for empty periods
        )
        return composite_image
    
    composites = ee.ImageCollection(list_of_dates.map(composite_for_millis))
    logging.info(f"Generated {composites.size().getInfo()} 8-day composites (including placeholders).")
    return composites


def download_ndvi(ndvi_composites, big_folder, roi, ndvi_name, roi_name, folder_name):
    """
    Downloads each NDVI image from the collection as a ZIP file into a temporary folder,
    unzips it to extract the GeoTIFF image, moves the TIFF to the destination folder,
    and removes the temporary folder afterwards. Includes retries for download.
    """
    image_list = ndvi_composites.toList(ndvi_composites.size())
    size = ndvi_composites.size().getInfo()
    out_folder = os.path.join(big_folder, roi_name, folder_name) # Ensure correct path structure
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        logging.info(f"Created output folder for downloads: {out_folder}")

    # logging.info(f"Starting download of {size} valid NDVI composites for ROI '{roi_name}'.")

    for i in range(size):
        image = ee.Image(image_list.get(i))
        
        # Ensure the image has an NDVI band and is not the exclude_date placeholder
        # Check against the actual value of exclude_date.getInfo()
        exclude_date_value = ee.Date('1900-01-01').millis().getInfo()
        if not image.bandNames().contains('NDVI').getInfo() or \
           image.get('system:time_start').getInfo() == exclude_date_value:
            
            logging.debug(f"Skipping empty or placeholder image at index {i}.")
            continue # Skip placeholder images

        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        # Construct the expected final file path
        final_tif_path = os.path.join(out_folder, f"{ndvi_name}{date_str}.tif")

        # Check if the file already exists
        if os.path.exists(final_tif_path):
            logging.info(f"Skipping download for {date_str}, file already exists: {final_tif_path}")
            continue

        params = {
            'scale': 10,
            'region': roi,
            'fileFormat': 'ZIP',  # Request a ZIP file from GEE
            'maxPixels': 1e13
        }
        download_url = image.select('NDVI').getDownloadURL(params)
        # logging.info(f'Attempting to download NDVI for date: {date_str}')
        # logging.debug(f'Download URL: {download_url}') # Log URL at debug level
        
        temp_dir = None
        max_retries = 3
        download_success = False

        for attempt in range(max_retries):
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, f"{ndvi_name}{date_str}.zip")
            
            try:
                # logging.info(f"Attempt {attempt + 1}/{max_retries} for {date_str}: Downloading ZIP to {zip_path}")
                response = requests.get(download_url, timeout=300) # Increased timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    # logging.info(f"Downloaded ZIP for {date_str}. Unzipping...")
                    
                    # Unzip the file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        tif_in_zip = [f for f in zip_ref.namelist() if f.lower().endswith('.tif')]
                        if tif_in_zip:
                            extracted_tif_name = tif_in_zip[0]
                            zip_ref.extract(extracted_tif_name, temp_dir)
                            extracted_tif_path = os.path.join(temp_dir, extracted_tif_name)
                            
                            shutil.move(extracted_tif_path, final_tif_path)
                            # logging.info(f"Successfully moved GeoTIFF for {date_str} to {final_tif_path}")
                            download_success = True
                            break # Break retry loop on success
                        else:
                            logging.warning(f"No TIFF file found in ZIP for {date_str}. Retrying...")
                else:
                    logging.warning(f"Download failed for {date_str} (Status code: {response.status_code}). Retrying...")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Request error for {date_str} (attempt {attempt+1}): {e}.")
            except zipfile.BadZipFile:
                logging.warning(f"Downloaded file is a bad ZIP for {date_str}. Retrying download...")
            except Exception as e:
                logging.error(f"Unhandled error during download or processing for {date_str} (attempt {attempt+1}): {e}", exc_info=True)
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logging.debug(f"Cleaned up temporary folder: {temp_dir}")
            
            if not download_success and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1)) # Exponential backoff before next retry
        
        if not download_success:
            logging.error(f"Failed to download and process NDVI for {date_str} after {max_retries} attempts.")


def main_ndvi(start_date, end_date, roi, roi_name, big_folder):
    """
    Main function to orchestrate NDVI retrieval and local storage.
    """

    logging.info(f"--- Starting NDVI retrieval process for ROI '{roi_name}' from {start_date} to {end_date} ---")
    
    # Define time period and region of interest.
    exclude_date = ee.Date('1900-01-01').millis() # Placeholder for empty composites
    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)

    # ---------------------------------------------------------------------------
    # Run processing steps
    # ---------------------------------------------------------------------------
    # 1. Load Sentinel-2 collection and mask clouds.
    try:
        sentinel_collection = get_sentinel_collection(start_date_ee, end_date_ee, roi)
    except Exception as e:
        logging.critical(f"Failed to get Sentinel-2 collection: {e}")
        return

    # 2. Separate the collection into cloud-free and cloudy subsets.
    collections = separate_collections(sentinel_collection)
    
    # 3. Create NDVI composites from the cloudy collection (as per original logic).
    # If the intention is to use the 'cloudFree' collection, change this line.
    ndvi_composites = calculate_8day_composites(collections['cloudy'], start_date_ee, end_date_ee, exclude_date)

    # Filter out empty placeholders before attempting download
    # This filter relies on the 'exclude_date' being set for empty composites
    valid_ndvi_composites = ndvi_composites.filter(ee.Filter.neq('system:time_start', exclude_date))
    valid_composites_count = valid_ndvi_composites.size().getInfo()
    logging.info(f'Found {valid_composites_count} valid 8-day NDVI composites for download (after removing empty placeholders).')

    if valid_composites_count == 0:
        logging.warning(f"No valid NDVI composites found for ROI '{roi_name}' in the period. Skipping download.")
        return

    # 4. Download NDVI composites to local storage.
    download_ndvi(valid_ndvi_composites, big_folder, roi, 'ndvi8days_', roi_name, f'{roi_name.split("_")[0]}_ndvi8days')

    logging.info(f"--- NDVI retrieval process finished for ROI '{roi_name}'. ---")


