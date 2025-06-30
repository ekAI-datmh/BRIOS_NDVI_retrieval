import ee
ee.Initialize(project='ee-hadat')

def addBand(landsat):
    """
    Returns a function that computes NDVI for a given Landsat image and adds it as a new band 'NDVI'.
    
    Parameters:
      landsat (str): Landsat satellite id (e.g., 'L4', 'L5', 'L7', 'L8', or 'L9')
      
    Returns:
      A function that takes an ee.Image and returns the image with an added NDVI band.
    """
    def wrap(image):
        # Choose bands based on the Landsat satellite.
        if landsat in ['L8', 'L9']:
            nir_band = 'SR_B5'
            red_band = 'SR_B4'
        else:
            nir_band = 'SR_B4'
            red_band = 'SR_B3'
        
        # Compute scaled reflectances for NIR and red bands.
        nir = image.select(nir_band).multiply(0.0000275).add(-0.2)
        red = image.select(red_band).multiply(0.0000275).add(-0.2)
        
        # Compute NDVI using the standard formula.
        ndvi = image.expression(
            '(nir - red) / (nir + red)',
            {'nir': nir, 'red': red}
        ).rename('NDVI')
        
        return image.addBands(ndvi)
    
    return wrap

# Example usage:
# ndvi_func = addBand('L8')
# image_with_ndvi = ndvi_func(ee.Image("LANDSAT/LC08/C02/T1_L2/LC08_044034_20170606"))
