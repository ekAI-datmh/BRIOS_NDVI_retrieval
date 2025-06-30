import ee
ee.Initialize(project='ee-hadat')

def addBand(landsat):
    """
    Returns a function that adds a 'FVC' band (fraction of vegetation cover)
    computed from the NDVI of the image.

    Parameters:
      landsat (str): Landsat satellite id (e.g., 'L4', 'L5', 'L7', 'L8').
                     (Currently not used in the calculation.)
    
    Returns:
      A function that takes an ee.Image and returns the image with an added
      'FVC' band.
    """
    def wrap(image):
        ndvi = image.select('NDVI')
        # Compute FVC using the provided expression.
        fvc = image.expression(
            '((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg))**2',
            {
                'ndvi': ndvi,
                'ndvi_bg': 0.2,
                'ndvi_vg': 0.86
            }
        )
        # Clamp FVC values between 0 and 1.
        fvc = fvc.where(fvc.lt(0.0), 0.0)
        fvc = fvc.where(fvc.gt(1.0), 1.0)
        return image.addBands(fvc.rename('FVC'))
    return wrap

# Example usage:
# fvc_func = addBand('L8')
# image_with_fvc = fvc_func(your_image)
