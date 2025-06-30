import ee
ee.Initialize(project='ee-hadat')

def toa(image):
    """
    Applies a cloud mask to TOA data using the QA_PIXEL band.
    
    Parameters:
      image (ee.Image): Input image for which clouds are to be masked.
    
    Returns:
      ee.Image: The input image with the cloud mask applied.
    """
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3)
    return image.updateMask(mask.Not())

def sr(image):
    """
    Applies a cloud and cloud shadow mask to Surface Reflectance (SR) data using the QA_PIXEL band.
    
    Parameters:
      image (ee.Image): Input image for which clouds and cloud shadows are to be masked.
    
    Returns:
      ee.Image: The input image with the cloud and shadow mask applied.
    """
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4))
    return image.updateMask(mask.Not())
