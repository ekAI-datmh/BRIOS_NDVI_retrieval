import ee
ee.Initialize(project='ee-hadat')

# Import the module with ASTER GED bare emissivity functions.
# It should provide functions: emiss_bare_band13(image) and emiss_bare_band14(image)
from lst_module import ASTER_bare_emiss as ASTERGED

def addBand(landsat, use_ndvi):
    """
    Computes the surface emissivity for a Landsat image using ASTER GED and FVC.
    
    Args:
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9').
      use_ndvi (bool): If True, apply dynamic emissivity (with NDVI-based vegetation correction);
                       if False, use emissivity derived directly from ASTER.
    
    Returns:
      A function that takes an ee.Image and returns the image with an added 'EM' band.
    """
    def wrap(image):
        # Define coefficients based on the Landsat satellite.
        if landsat == 'L4':
            c13 = 0.3222
            c14 = 0.6498
            c    = 0.0272
        elif landsat == 'L5':
            c13 = -0.0723
            c14 = 1.0521
            c    = 0.0195
        elif landsat == 'L7':
            c13 = 0.2147
            c14 = 0.7789
            c    = 0.0059
        else:  # For L8, L9, etc.
            c13 = 0.6820
            c14 = 0.2578
            c    = 0.0584

        # Compute ASTER-based bare emissivity, convolved to the Landsat band.
        emiss_bare = image.expression(
            'c13 * EM13 + c14 * EM14 + c',
            {
                'EM13': ASTERGED.emiss_bare_band13(image),
                'EM14': ASTERGED.emiss_bare_band14(image),
                'c13': ee.Image(c13),
                'c14': ee.Image(c14),
                'c': ee.Image(c)
            }
        )
        
        # Compute dynamic emissivity using vegetation cover correction (FVC).
        EMd = image.expression(
            'fvc * 0.99 + (1 - fvc) * em_bare',
            {
                'fvc': image.select('FVC'),
                'em_bare': emiss_bare
            }
        )
        
        # Compute emissivity directly from ASTER without vegetation correction.
        aster = ee.Image("NASA/ASTER_GED/AG100_003").clip(image.geometry())
        EM0 = image.expression(
            'c13 * EM13 + c14 * EM14 + c',
            {
                'EM13': aster.select('emissivity_band13').multiply(0.001),
                'EM14': aster.select('emissivity_band14').multiply(0.001),
                'c13': ee.Image(c13),
                'c14': ee.Image(c14),
                'c': ee.Image(c)
            }
        )
        
        # Select which emissivity to use based on the use_ndvi flag.
        EM = ee.Image(ee.Algorithms.If(use_ndvi, EMd, EM0))
        
        # Prescribe emissivity values for water bodies and snow/ice.
        qa = image.select('QA_PIXEL')
        EM = EM.where(qa.bitwiseAnd(1 << 7), 0.99)
        EM = EM.where(qa.bitwiseAnd(1 << 5), 0.989)
        
        return image.addBands(EM.rename('EM'))
    
    return wrap

# Example usage:
# landsat_id = 'L8'
# use_ndvi = True
# em_func = addBand(landsat_id, use_ndvi)
# image_with_em = em_func(ee.Image("YOUR_IMAGE_ID_HERE"))
