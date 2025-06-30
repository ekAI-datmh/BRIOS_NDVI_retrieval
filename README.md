# BRIOS NDVI/RVI Data Retrieval

This project contains scripts to download NDVI (Normalized Difference Vegetation Index) and RVI (Radar Vegetation Index) data from Google Earth Engine for specified Regions of Interest (ROIs).

## Setup

1.  **Install Dependencies:**
    Before running the scripts, install the required Python packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Google Earth Engine Authentication:**
    Ensure you have authenticated with Google Earth Engine. If this is your first time, run the following command in your terminal and follow the on-screen instructions:

    ```bash
    earthengine authenticate
    ```

## Configuration

All key parameters are located in the `HYPERPARAMETERS & CONFIGURATION` section at the bottom of `main.py`.

Before running the script, please configure the following:

-   **File Paths:**
    -   `excel_file_path`: Set the path to your Excel file containing the ROI coordinates.
    -   `big_folder`: Define the main output directory where all downloaded data will be stored.

-   **Crawling Control:**
    -   `CRAWL_RVI`: Set to `True` to download RVI data, or `False` to skip.
    -   `CRAWL_NDVI`: Set to `True` to download NDVI data, or `False` to skip.

-   **ROI Partitioning:**
    -   To manage large datasets, you can process a subset of your ROIs by setting `ROI_SLICE_START` and `ROI_SLICE_END`.
    -   For example, to process the first 50 ROIs, set `ROI_SLICE_START = 0` and `ROI_SLICE_END = 50`. To process the next batch, you could set it to `ROI_SLICE_START = 50` and `ROI_SLICE_END = 100`.
    -   To process all ROIs, set `ROI_SLICE_START = 0` and `ROI_SLICE_END = None`.

## Usage

After configuring `main.py`, run the script from your terminal:

```bash
python main.py
```

The script will log its progress to the console and save a detailed log in `data_retrieval.log`.
