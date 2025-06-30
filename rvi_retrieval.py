import ee, math, os
import requests
import shutil
import time
import zipfile
import rasterio
import numpy as np
from rasterio.merge import merge
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
# ee.Authenticate(force = True)
# ee.Initialize(project='ee-hadat-461702-p4')

# Configure logging for RVI module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe rate limiter for downloads
download_lock = threading.Lock()
last_download_time = 0
MIN_DOWNLOAD_INTERVAL = 1.5  # Minimum seconds between downloads

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

# --- Helper Functions ---

def has_data_in_roi(image, roi):
    bands = ['VV', 'VH', 'RVI']
    for band in bands:
        if band not in image.bandNames().getInfo():
            return False
        stats = image.select(band).reduceRegion(ee.Reducer.minMax(), geometry=roi, scale=10)
        min_val = stats.get(band + '_min')
        if min_val is None:
            return False
    return True

def edge_correction(image):
    """
    Applies an edge correction to remove pixels with no data along the image boundary.
    """
    edge_mask = image.mask().reduce(ee.Reducer.allNonZero())
    return image.updateMask(edge_mask).copyProperties(image, ['system:time_start'])

def get_sentinel1_collection(start_date, end_date, roi):
    """
    Loads and clips the Sentinel-1 GRD collection filtered by date, region, and polarization.
    """
    s_date = start_date.advance(-8, 'day')
    e_date = end_date.advance(8, 'day')
    
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(s_date, e_date)
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    def process(image):
        image = image.clip(roi)
        image = edge_correction(image)
        return image

    return collection.map(process)

def calculate_rvi(image):
    """
    Calculates the Radar Vegetation Index (RVI) from the VV and VH bands.
    Formula: RVI = (4 * VH) / (VV + VH)
    """
    vv = image.select('VV')
    vh = image.select('VH')
    rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
    return image.addBands(rvi).set('system:time_start', image.get('system:time_start'))

def calculate_8day_composites_sar(image_collection, start_date, end_date):
    days_step = 8
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    millis_step = days_step * 24 * 60 * 60 * 1000
    list_of_dates = ee.List.sequence(start.millis(), end.millis(), millis_step)

    def composite_for_millis(millis):
        composite_center = ee.Date(millis)
        composite_start = composite_center.advance(-int(days_step/2), 'day')
        composite_end = composite_center.advance(int(days_step/2), 'day')
        period_collection = image_collection.filterDate(composite_start, composite_end)
        # Capture original image dates for mapping
        original_dates = period_collection.aggregate_array('system:time_start')
        # Create composite and attach original_dates property
        composite_image = ee.Image(ee.Algorithms.If(
            period_collection.size().gt(0),
            calculate_rvi(period_collection.median())
                .set('system:time_start', composite_center.millis())
                .set('original_dates', original_dates),
            ee.Image(0)
                .updateMask(ee.Image(0))
                .set('system:time_start', composite_center.millis())
                .set('original_dates', ee.List([]))
        ))
        return composite_image

    composites = ee.ImageCollection(list_of_dates.map(composite_for_millis))
    return composites

def sort_by_time(composites):
    return composites.sort('system:time_start')

def smooth_time_series(composites):
    """Apply smoothing to time series with error handling to prevent hanging"""
    try:
        logger.info("Starting time series smoothing...")
        image_list = composites.toList(composites.size())
        collection_size = composites.size().getInfo()
        
        logger.info(f"Smoothing {collection_size} images in time series")

        def has_rvi(img):
            return ee.Number(img.bandNames().size()).gt(3)

        smoothed_images = []
        
        # Process in smaller batches to avoid timeouts
        batch_size = 20
        for batch_start in range(0, collection_size, batch_size):
            batch_end = min(batch_start + batch_size, collection_size)
            logger.info(f"Processing smoothing batch {batch_start//batch_size + 1}: images {batch_start}-{batch_end}")
            
            for i in range(batch_start, batch_end):
                try:
                    image = ee.Image(image_list.get(i))
                    image_date = ee.Date(image.get('system:time_start'))
                    previous = ee.Image(image_list.get(i - 1)) if i > 0 else image
                    next_img = ee.Image(image_list.get(i + 1)) if i < (collection_size - 1) else image

                    current_has = has_rvi(image)
                    previous_has = has_rvi(previous)
                    next_has = has_rvi(next_img)

                    smoothed = ee.Image(ee.Algorithms.If(
                        current_has,
                        ee.Image(ee.Algorithms.If(
                            previous_has.And(next_has),
                            ee.ImageCollection([previous, image, next_img]).mean().set('system:time_start', image_date.millis()),
                            ee.Image(ee.Algorithms.If(
                                next_has,
                                ee.ImageCollection([image, next_img]).mean().set('system:time_start', image_date.millis()),
                                ee.Image(ee.Algorithms.If(
                                    previous_has,
                                    ee.ImageCollection([image, previous]).mean().set('system:time_start', image_date.millis()),
                                    image
                                ))
                            ))
                        )),
                        ee.Image(ee.Algorithms.If(
                            previous_has.And(next_has),
                            ee.ImageCollection([previous, next_img]).mean().set('system:time_start', image_date.millis()),
                            ee.Image(ee.Algorithms.If(
                                previous_has,
                                previous.set('system:time_start', image_date.millis()),
                                ee.Image(ee.Algorithms.If(
                                    next_has,
                                    next_img.set('system:time_start', image_date.millis()),
                                    image
                                ))
                            ))
                        ))
                    ))
                    smoothed_images.append(smoothed)
                    
                except Exception as e:
                    logger.warning(f"Error smoothing image {i}: {e}")
                    # Add the original image if smoothing fails
                    try:
                        original_image = ee.Image(image_list.get(i))
                        smoothed_images.append(original_image)
                    except:
                        logger.error(f"Failed to add original image {i}")
                        continue
            
            # Small delay between batches
            time.sleep(0.5)

        logger.info(f"Completed time series smoothing for {len(smoothed_images)} images")
        return ee.ImageCollection(smoothed_images)
        
    except Exception as e:
        logger.error(f"Error in time series smoothing: {e}")
        logger.info("Returning original composites without smoothing")
        return composites

def download_band(image, band_name, roi, date_str, out_folder, max_retries=4):
    """Download a single band with rate limiting and improved error handling"""
    rate_limited_download()  # Apply rate limiting
    
    local_path = os.path.join(out_folder, f"{band_name}_{date_str}.tif")
    temp_dir = os.path.join(out_folder, "temp", f"{band_name}_{date_str}")
    os.makedirs(temp_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            url = image.select([band_name]).getDownloadURL({
                'scale': 10,
                'region': roi,
                'fileFormat': 'GeoTIFF',
                'maxPixels': 1e13,
                'expires': 3600
            })
            logger.debug(f"Attempt {attempt+1} for {band_name} {date_str}: {url}")

            temp_zip_path = os.path.join(temp_dir, "download.zip")
            response = requests.get(url, stream=True, timeout=300, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
            logger.debug(f"Status code: {response.status_code}")

            response.raise_for_status()

            with open(temp_zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                logger.debug(f"Zip file contents: {zip_ref.namelist()}")
                zip_ref.extractall(temp_dir)

            logger.debug(f"Files in temp dir: {os.listdir(temp_dir)}")
            tif_files = [f for f in os.listdir(temp_dir) if f.endswith('.tif')]

            if len(tif_files) == 1:
                tif_file = tif_files[0]
                src_path = os.path.join(temp_dir, tif_file)
                shutil.copy(src_path, local_path)
                if is_valid_tif(local_path):
                    logger.info(f"Successfully downloaded {band_name} for {date_str}")
                    # Clean up temp directory after successful download
                    shutil.rmtree(temp_dir)
                    return local_path
                else:
                    logger.warning(f"Invalid file for {band_name} {date_str}, retrying...")
                    time.sleep(1 * (attempt + 1))
            else:
                logger.warning(f"Unexpected number of .tif files in zip: {len(tif_files)}")
                time.sleep(1 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {band_name} {date_str}: {e}")
            time.sleep(1 * (attempt + 1))
        except Exception as e:
            logger.error(f"Error downloading {band_name} for {date_str}: {e}")
            time.sleep(1 * (attempt + 1))
    
    logger.error(f"Failed to download {band_name} for {date_str} after {max_retries} attempts.")
    # Clean up temp directory on failure
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    return None

def merge_bands(rvi_path, vv_path, vh_path, output_path):
    """Merge RVI, VV, VH into a single 3-band image (reshaping if necessary)."""
    band_paths = [rvi_path, vv_path, vh_path]
    bands = []

    for path in band_paths:
        with rasterio.open(path) as src:
            band = src.read(1)
            bands.append((band, src.profile))

    min_shape = min(band[0].shape for band in bands)
    bands_resized = [np.resize(band[0], min_shape) for band in bands]

    profile = bands[0][1]
    profile.update(count=3, dtype='float32')

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, band in enumerate(bands_resized, start=1):
            dst.write(band, i)

    print(f"✅ Merged 3-band image saved: {output_path}")

def is_valid_tif(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        print(f"Invalid: {file_path} - File missing or too small")
        return False
    try:
        with rasterio.open(file_path) as src:
            if src.count == 0 or src.width == 0 or src.height == 0:
                print(f"Invalid: {file_path} - No bands or zero dimensions")
                return False
            return True
    except rasterio.errors.RasterioIOError as e:
        print(f"Invalid: {file_path} - Rasterio error: {e}")
        return False

def download_bands_parallel(image, roi, date_str, out_folder, bands=['RVI', 'VV', 'VH'], max_workers=2):
    """
    Download multiple bands in parallel for a single image.
    
    Args:
        image: Earth Engine image
        roi: Region of interest
        date_str: Date string for naming
        out_folder: Output folder
        bands: List of band names to download
        max_workers: Maximum number of concurrent downloads (default: 2)
    
    Returns:
        dict: Dictionary mapping band names to downloaded file paths (None if failed)
    """
    results = {}
    
    # Check if merged file already exists
    final_merged_tif_path = os.path.join(out_folder, f"rvi_8days_{date_str}.tif")
    if os.path.exists(final_merged_tif_path):
        logger.info(f"Skipping download for {date_str}, merged file already exists")
        return {'merged_file': final_merged_tif_path}
    
    if not has_data_in_roi(image, roi):
        logger.info(f"No data for {date_str}, skipping.")
        return {}
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_band = {
            executor.submit(download_band, image, band, roi, date_str, out_folder): band
            for band in bands
        }
        
        # Collect results
        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                result = future.result()
                results[band] = result
                if result:
                    logger.debug(f"Successfully downloaded {band} for {date_str}")
                else:
                    logger.warning(f"Failed to download {band} for {date_str}")
            except Exception as exc:
                logger.error(f"Exception downloading {band} for {date_str}: {exc}")
                results[band] = None
    
    return results

def export_sentinel1_rvi(sentinel_collection, big_folder, roi, image_name, roi_name, folder_name, enable_parallel=True):
    """Export Sentinel-1 RVI data with optional parallel processing for band downloads"""
    out_folder = os.path.join(big_folder, roi_name, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    image_list = sentinel_collection.toList(sentinel_collection.size())
    count = sentinel_collection.size().getInfo()
    
    logger.info(f"Starting export of {count} images for {roi_name}")

    successful_downloads = 0
    failed_downloads = 0

    for i in range(count):
        image = ee.Image(image_list.get(i))
        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        try:
            if enable_parallel:
                # Use parallel downloading
                download_results = download_bands_parallel(image, roi, date_str, out_folder)
                
                if 'merged_file' in download_results:
                    # File already exists, skip
                    continue
                
                # Check if all bands were downloaded successfully
                rvi_path = download_results.get('RVI')
                vv_path = download_results.get('VV')
                vh_path = download_results.get('VH')
                
            else:
                # Sequential downloading (original approach)
                final_merged_tif_path = os.path.join(out_folder, f"{image_name}_{date_str}.tif")
                if os.path.exists(final_merged_tif_path):
                    logger.info(f"Skipping download for {date_str}, merged file already exists")
                    continue
                
                if not has_data_in_roi(image, roi):
                    logger.info(f"No data for {date_str}, skipping.")
                    continue

                rvi_path = download_band(image, 'RVI', roi, date_str, out_folder)
                vv_path = download_band(image, 'VV', roi, date_str, out_folder)
                vh_path = download_band(image, 'VH', roi, date_str, out_folder)

            # Merge bands if all were downloaded successfully
            if rvi_path and vv_path and vh_path:
                output_path = os.path.join(out_folder, f"{image_name}_{date_str}.tif")
                merge_bands(rvi_path, vv_path, vh_path, output_path)
                successful_downloads += 1
                
                # Remove individual band files after successful merge
                for band_path in [rvi_path, vv_path, vh_path]:
                    if os.path.exists(band_path):
                        os.remove(band_path)
                        logger.debug(f"Removed individual band file: {band_path}")
            else:
                logger.warning(f"Skipping merge for {date_str} due to missing or invalid bands.")
                failed_downloads += 1
                
                # Clean up any downloaded files if merge fails
                for band_path in [rvi_path, vv_path, vh_path]:
                    if band_path and os.path.exists(band_path):
                        os.remove(band_path)
                        logger.debug(f"Removed partial band file: {band_path}")
                        
        except Exception as e:
            logger.error(f"Error processing image {date_str}: {e}")
            failed_downloads += 1

    logger.info(f"Export completed for {roi_name}: {successful_downloads} successful, {failed_downloads} failed")

def display_rvi(rvi_collection, layer_name):
    image_list = rvi_collection.toList(rvi_collection.size())
    count = rvi_collection.size().getInfo()
    for i in range(count):
        rvi_image = ee.Image(image_list.get(i))
        if rvi_image.bandNames().contains('RVI').getInfo():
            date = ee.Date(rvi_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            print(f"{layer_name} - Image with RVI from {date}")

# def export_sentinel1_rvi_drive(sentinel_collection, roi, image_name, folder_name):
#     image_list = sentinel_collection.toList(sentinel_collection.size())
#     count = sentinel_collection.size().getInfo()
#     for i in range(count):
#         image = ee.Image(image_list.get(i))
#         date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
#         task = ee.batch.Export.image.toDrive(
#             image=image.select(['RVI', 'VV', 'VH']),
#             description=image_name + date_str,
#             folder=folder_name,
#             scale=10,
#             region=roi,
#             fileFormat='GeoTIFF',
#             maxPixels=1e13
#         )
#         task.start()
#         print('Export initiated for image on date:', date_str)

def generate_rvi_date_mapping(composites, big_folder, roi_name):
    """
    Generate and save a JSON mapping of original RVI image dates to
    their nearest composite time-series dates.
    """
    try:
        logger.info(f"Generating RVI date mapping for {roi_name}...")
        
        # Check if mapping already exists
        out_folder = os.path.join(big_folder, roi_name)
        os.makedirs(out_folder, exist_ok=True)
        json_path = os.path.join(out_folder, f"{roi_name}_rvi_date_mapping.json")
        
        if os.path.exists(json_path):
            logger.info(f"RVI date mapping already exists: {json_path}")
            return {}
        
        mapping_candidates = {}
        size = composites.size().getInfo()
        
        # Limit processing to avoid hanging - only process first 50 composites for mapping
        max_composites = min(size, 50)
        logger.info(f"Processing {max_composites} out of {size} composites for date mapping")
        
        images_list = composites.limit(max_composites).toList(max_composites)
        
        for i in range(max_composites):
            try:
                # Add rate limiting for GEE API calls
                if i > 0 and i % 10 == 0:
                    logger.info(f"Processed {i}/{max_composites} composites for date mapping")
                    time.sleep(1)  # Brief pause every 10 iterations
                
                image = ee.Image(images_list.get(i))
                comp_millis = image.get('system:time_start').getInfo()
                original_millis_list = image.get('original_dates').getInfo()
                
                # Handle case where original_dates might be null or empty
                if original_millis_list and isinstance(original_millis_list, list):
                    for orig in original_millis_list:
                        if orig:  # Check for null values
                            mapping_candidates.setdefault(orig, []).append(comp_millis)
                            
            except Exception as e:
                logger.warning(f"Error processing composite {i} for date mapping: {e}")
                continue
        
        # Select nearest composite date for each original date
        mapping = {}
        for orig, comps in mapping_candidates.items():
            try:
                closest = min(comps, key=lambda c: abs(c - orig))
                orig_date = datetime.datetime.utcfromtimestamp(orig/1000).strftime('%Y-%m-%d')
                comp_date = datetime.datetime.utcfromtimestamp(closest/1000).strftime('%Y-%m-%d')
                mapping[orig_date] = comp_date
            except Exception as e:
                logger.warning(f"Error processing date mapping for {orig}: {e}")
                continue
        
        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Saved RVI date mapping with {len(mapping)} entries to {json_path}")
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to generate RVI date mapping for {roi_name}: {e}")
        return {}

def main_rvi(start_date, end_date, ROI, big_folder, roi_name, enable_parallel=True, generate_date_mapping=False):
    """Main RVI processing function with optional parallel processing"""
    logger.info(f"Starting RVI processing for {roi_name} from {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
    
    sentinel1_collection = get_sentinel1_collection(start_date, end_date, ROI)
    logger.info(f'Sentinel-1 collection size: {sentinel1_collection.size().getInfo()}')

    sentinel_8day_composites = calculate_8day_composites_sar(sentinel1_collection, start_date, end_date)
    logger.info(f'8-day composites size: {sentinel_8day_composites.size().getInfo()}')

    # Generate JSON mapping original RVI dates to composite time-series dates (optional)
    if generate_date_mapping:
        generate_rvi_date_mapping(sentinel_8day_composites, big_folder, roi_name)
    else:
        logger.info("Skipping RVI date mapping generation to avoid potential hanging")

    sorted_composites = sort_by_time(sentinel_8day_composites)
    logger.info(f'First composite band names: {sorted_composites.first().bandNames().getInfo()}')

    smoothed_composites = smooth_time_series(sorted_composites)

    # Use parallel processing for downloads
    export_sentinel1_rvi(
        smoothed_composites, 
        big_folder, 
        ROI, 
        'rvi_8days', 
        roi_name, 
        f'{roi_name.split("_")[0]}_rvi_8days',
        enable_parallel=enable_parallel
    )
    
    logger.info(f"Completed RVI processing for {roi_name}")
    # export_sentinel1_rvi_drive(smoothed_composites, ROI, 'rvi_8days_', 'LST')

# # --- Main Execution ---

# start_date = ee.Date('2023-01-01')
# end_date = ee.Date('2023-02-01')
# ROI = ee.Geometry.Polygon([
#     [[11855180.0, 2328782.140517925], [11855180.0, 2335016.8225009246], [11851160.0, 2335016.8225009246], [11851160.0, 2328782.140517925], [11855180.0, 2328782.140517925]]
# ], 'EPSG:3857')

# big_folder = "/mnt/data1tb/LSTRetrieval/Code/download_data"

# sentinel1_collection = get_sentinel1_collection(start_date, end_date, ROI)
# print('Sentinel-1 collection size:', sentinel1_collection.size().getInfo())

# sentinel_8day_composites = calculate_8day_composites_sar(sentinel1_collection, start_date, end_date)
# print('8-day composites size:', sentinel_8day_composites.size().getInfo())

# sorted_composites = sort_by_time(sentinel_8day_composites)
# print('First composite band names:', sorted_composites.first().bandNames().getInfo())

# smoothed_composites = smooth_time_series(sorted_composites)

# export_sentinel1_rvi(smoothed_composites, big_folder, ROI, 'rvi_8days', "BinhThanh_DucHue_LongAn", 'BinhThanh_rvi_8days')
# # export_sentinel1_rvi_drive(smoothed_composites, ROI, 'rvi_8days_', 'LST')