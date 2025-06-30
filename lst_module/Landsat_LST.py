import ee
ee.Initialize(project='ee-hadat')

# Assume that the following modules have been ported to Python.
# Their functions should mimic the JS modules:
#
#   NCEP_TPW.addBand(image)
#   cloudmask.toa(image)
#   cloudmask.sr(image)
#   NDVI.addBand(image, landsat)
#   FVC.addBand(image, landsat)
#   EM.addBand(image, landsat, use_ndvi)
#   LST.addBand(image, landsat)
#
# You can import them as follows (adjust import paths as needed):
from lst_module import NCEP_TPW
from lst_module import cloudmask
from lst_module import compute_NDVI as NDVI
from lst_module import compute_FVC as FVC
from lst_module import compute_emissivity as EM
from lst_module import SMWalgorithm as LST

# Define the dictionary for Landsat collections.
COLLECTION = ee.Dictionary({
    'L4': {
        'TOA': ee.ImageCollection('LANDSAT/LT04/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LT04/C02/T1_L2'),
        'TIR': ['B6'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L5': {
        'TOA': ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'),
        'TIR': ['B6'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L7': {
        'TOA': ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
        'TIR': ['B6_VCID_1', 'B6_VCID_2'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L8': {
        'TOA': ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    },
    'L9': {
        'TOA': ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    }
})

def collection(landsat, date_start, date_end, geometry, use_ndvi):
    """
    Selects the Landsat data based on user inputs and performs the LST computation.
    
    INPUTS:
      landsat: string, one of 'L4', 'L5', 'L7', 'L8', or 'L9'
      date_start: string in 'YYYY-MM-DD' format (start date)
      date_end: string in 'YYYY-MM-DD' format (end date)
      geometry: ee.Geometry defining the region of interest
      use_ndvi: boolean, if True, NDVI is used to obtain dynamic emissivity; otherwise,
                emissivity is obtained directly from ASTER.
    
    OUTPUT:
      An ee.ImageCollection with the following bands:
          - Original Landsat bands (from SR collection except TIR bands)
          - TIR bands (from the TOA collection)
          - 'NDVI': normalized vegetation index
          - 'FVC': fraction of vegetation cover [0-1]
          - 'TPW': total precipitable water [mm]
          - 'EM': surface emissivity for TIR band
          - 'LST': land surface temperature
    """
    # Get the collection dictionary for the requested Landsat satellite.
    collection_dict = ee.Dictionary(COLLECTION.get(landsat))
    
    # Load the TOA (Top-of-Atmosphere) Radiance/Reflectance collection.
    landsatTOA = ee.ImageCollection(collection_dict.get('TOA')) \
        .filterDate(date_start, date_end) \
        .filterBounds(geometry)
        # .map(cloudmask.toa)  # Optionally, apply the TOA cloud mask if needed.
    
    # Load the Surface Reflectance collection.
    landsatSR = ee.ImageCollection(collection_dict.get('SR')) \
        .filterDate(date_start, date_end) \
        .filterBounds(geometry)
    
    # Process the Surface Reflectance collection by adding extra bands.
    landsatSR = landsatSR.map(NDVI.addBand(landsat)) \
                         .map(cloudmask.sr) \
                         .map(FVC.addBand(landsat)) \
                         .map(NCEP_TPW.addBand) \
                         .map(EM.addBand(landsat, use_ndvi))
    
    # Define the TIR and visible/shortwave bands to keep.
    tir = ee.List(collection_dict.get('TIR'))
    visw = ee.List(collection_dict.get('VISW')) \
             .cat(ee.List(['NDVI', 'FVC', 'TPW', 'TPWpos', 'EM']))
    
    # Combine the collections:
    # - Use all channels from the SR collection (except TIR bands)
    # - Merge in the TIR bands from the TOA collection.
    landsatALL = landsatSR.select(visw) \
                    .combine(landsatTOA.select(tir), overwrite=True)
    
    # Compute the LST by mapping the LST algorithm over the combined collection.
    landsatLST = landsatALL.map(LST.addBand(landsat))
    
    return landsatLST

# Example usage:
# Define input parameters.
# landsat_id = 'L8'
# date_start = '2020-01-01'
# date_end = '2020-12-31'
# geometry = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])  # Replace with your ROI.
# use_ndvi = True
#
# landsat_lst_collection = collection(landsat_id, date_start, date_end, geometry, use_ndvi)
# print(landsat_lst_collection.getInfo())
