import ee
ee.Initialize(project='ee-hadat')

# Import the ASTER GED bare emissivity module.
# It should provide functions: emiss_bare_band10, emiss_bare_band11, emiss_bare_band12,
# emiss_bare_band13, and emiss_bare_band14.
from lst_module import ASTER_bare_emiss  

def addBand(dynamic):
    """
    Computes broad-band emissivity (BBE) from ASTER GED and adds it as a band ('BBE')
    to the input image. If dynamic is True, vegetation cover correction is applied;
    otherwise, the original ASTER GED emissivity is used.
    
    Parameters:
      dynamic (bool): True to use vegetation cover correction, False to use original emissivity.
    
    Returns:
      A function that takes an ee.Image and returns an ee.Image with an added 'BBE' band.
    """
    def wrap(image):
        # Get ASTER emissivity image and clip to the image geometry.
        aster = ee.Image("NASA/ASTER_GED/AG100_003").clip(image.geometry())
        
        # Process emissivity band 10.
        orig = aster.select('emissivity_band10').multiply(0.001)
        dynam = image.expression('fvc*0.99+(1-fvc)*em_bare', {
            'fvc': image.select('FVC'),
            'em_bare': ASTER_bare_emiss.emiss_bare_band10(image)
        })
        em10 = ee.Image(ee.Algorithms.If(dynamic, dynam, orig))
        
        # Process emissivity band 11.
        orig = aster.select('emissivity_band11').multiply(0.001)
        dynam = image.expression('fvc*0.99+(1-fvc)*em_bare', {
            'fvc': image.select('FVC'),
            'em_bare': ASTER_bare_emiss.emiss_bare_band11(image)
        })
        em11 = ee.Image(ee.Algorithms.If(dynamic, dynam, orig))
        
        # Process emissivity band 12.
        orig = aster.select('emissivity_band12').multiply(0.001)
        dynam = image.expression('fvc*0.99+(1-fvc)*em_bare', {
            'fvc': image.select('FVC'),
            'em_bare': ASTER_bare_emiss.emiss_bare_band12(image)
        })
        em12 = ee.Image(ee.Algorithms.If(dynamic, dynam, orig))
        
        # Process emissivity band 13.
        orig = aster.select('emissivity_band13').multiply(0.001)
        dynam = image.expression('fvc*0.99+(1-fvc)*em_bare', {
            'fvc': image.select('FVC'),
            'em_bare': ASTER_bare_emiss.emiss_bare_band13(image)
        })
        em13 = ee.Image(ee.Algorithms.If(dynamic, dynam, orig))
        
        # Process emissivity band 14.
        orig = aster.select('emissivity_band14').multiply(0.001)
        dynam = image.expression('fvc*0.99+(1-fvc)*em_bare', {
            'fvc': image.select('FVC'),
            'em_bare': ASTER_bare_emiss.emiss_bare_band14(image)
        })
        em14 = ee.Image(ee.Algorithms.If(dynamic, dynam, orig))
        
        # Compute the broad-band emissivity (BBE) using the linear combination.
        bbe = image.expression(
            '0.128 + 0.014*em10 + 0.145*em11 + 0.241*em12 + 0.467*em13 + 0.004*em14',
            {
                'em10': em10,
                'em11': em11,
                'em12': em12,
                'em13': em13,
                'em14': em14
            }
        )
        
        return image.addBands(bbe.rename('BBE'))
    
    return wrap

# Example usage:
# dynamic = True  # Use vegetation cover correction.
# bbe_func = addBand(dynamic)
# image_with_bbe = bbe_func(ee.Image("YOUR_IMAGE_ID_HERE"))
