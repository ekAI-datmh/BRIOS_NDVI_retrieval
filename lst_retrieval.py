import ee
import requests  # used to download files
import os
import tempfile
import zipfile
import shutil
import time
# ee.Authenticate(force = True)
# ee.Initialize(project='ee-hadat-461702-p4')

# Import the Landsat LST computation module.
from lst_module import Landsat_LST as LandsatLST


def lst_retrive(date_start, date_end, geometry, ROI, main_folder):
    satellites = ["L8", "L9"]

    for satellite in satellites:
        # Define parameters.
        # date_start = '2022-12-15'
        # date_end = '2023-01-01'
        use_ndvi = True

        # Get Landsat collection with added variables.
        LandsatColl = LandsatLST.collection(satellite, date_start, date_end, geometry, use_ndvi)
        print('Landsat Collection:', LandsatColl.getInfo())

        # Convert the collection to a list.
        imageList = LandsatColl.toList(LandsatColl.size())
        imageCount = LandsatColl.size().getInfo()
        print('Number of images to process:', imageCount)
        print()

        for i in range(imageCount):
            image = ee.Image(imageList.get(i))
            # Get the image date formatted as YYYY-MM-dd.
            imageDate = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            # Define destination path and check if file already exists
            dest_folder_path = os.path.join(main_folder, ROI, "lst")
            expected_dest_tif_path = os.path.join(dest_folder_path, f"lst16days_{imageDate}.tif")

            if os.path.exists(expected_dest_tif_path):
                print(f"Skipping download for {imageDate}, file already exists: {expected_dest_tif_path}")
                print("\n")
                time.sleep(0.5) # Shorter pause if skipped
                continue

            # Get a download URL for the LST band as a ZIP.
            download_params = {
                'scale': 30,
                'region': geometry,  # geometry can be passed directly if it is an ee.Geometry
                'fileFormat': 'ZIP'
            }
            download_url = image.select('LST').getDownloadURL(download_params)
            print('Downloading image for date:', imageDate)
            # print('Download URL:', download_url)

            # Create a temporary folder.
            temp_dir = tempfile.mkdtemp()
            zip_filename = os.path.join(temp_dir, f"lst16days_{imageDate}.zip")
            
            try:
                response = requests.get(download_url, timeout=120)
                if response.status_code == 200:
                    # Save the ZIP file.
                    with open(zip_filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded ZIP file to {zip_filename}")
                    
                    # Extract the ZIP file.
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    print(f"Extracted ZIP file in {temp_dir}")
                    
                    # Look for the first .tif file in the temporary folder.
                    tif_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.tif')]
                    if tif_files:
                        tif_file = tif_files[0]
                        source_tif = os.path.join(temp_dir, tif_file)
                        # Ensure the destination folder exists.
                        dest_folder = os.path.join(main_folder, ROI)
                        dest_folder = os.path.join(dest_folder, "lst")
                        os.makedirs(dest_folder, exist_ok=True)
                        dest_tif = os.path.join(dest_folder, f"lst16days_{imageDate}.tif")
                        shutil.move(source_tif, dest_tif)
                        print(f"Moved extracted TIFF to {dest_tif}")
                    else:
                        print(f"No TIFF file found in ZIP for date: {imageDate}")
                else:
                    print(f"Failed to download image for date: {imageDate}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Exception occurred for {imageDate}: {e}")
            finally:
                # Remove the temporary folder regardless of outcome.
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Removed temporary folder: {temp_dir}")
                except Exception as e:
                    print(f"Error removing temporary folder {temp_dir}: {e}")
            print("\n")
            time.sleep(2)  # Pause briefly between downloads if necessary

# Run the function
# lst_retrive(date_start, date_end, geometry, "AnNinh-QuynhPhu-ThaiBinh", "/mnt/data1tb/LSTRetrieval/Code/download_data")
