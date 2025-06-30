import ee
ee.Initialize(project='ee-hadat')

# Get ASTER emissivity image.
aster = ee.Image("NASA/ASTER_GED/AG100_003")

# Get ASTER NDVI and convert to NDVI scale.
aster_ndvi = aster.select('ndvi').multiply(0.01)

# Compute ASTER FVC from NDVI.
aster_fvc = aster_ndvi.expression(
    '((ndvi - ndvi_bg)/(ndvi_vg - ndvi_bg))**2',
    {
        'ndvi': aster_ndvi,
        'ndvi_bg': 0.2,
        'ndvi_vg': 0.86
    }
)

# Clamp FVC values to the [0, 1] range.
aster_fvc = aster_fvc.where(aster_fvc.lt(0.0), 0.0)
aster_fvc = aster_fvc.where(aster_fvc.gt(1.0), 1.0)

# Bare ground emissivity function for band 10.
def emiss_bare_band10(image):
    return image.expression(
        '(EM - 0.99*fvc)/(1.0 - fvc)',
        {
            'EM': aster.select('emissivity_band10').multiply(0.001),
            'fvc': aster_fvc
        }
    ).clip(image.geometry())

# Bare ground emissivity function for band 11.
def emiss_bare_band11(image):
    return image.expression(
        '(EM - 0.99*fvc)/(1.0 - fvc)',
        {
            'EM': aster.select('emissivity_band11').multiply(0.001),
            'fvc': aster_fvc
        }
    ).clip(image.geometry())

# Bare ground emissivity function for band 12.
def emiss_bare_band12(image):
    return image.expression(
        '(EM - 0.99*fvc)/(1.0 - fvc)',
        {
            'EM': aster.select('emissivity_band12').multiply(0.001),
            'fvc': aster_fvc
        }
    ).clip(image.geometry())

# Bare ground emissivity function for band 13.
def emiss_bare_band13(image):
    return image.expression(
        '(EM - 0.99*fvc)/(1.0 - fvc)',
        {
            'EM': aster.select('emissivity_band13').multiply(0.001),
            'fvc': aster_fvc
        }
    ).clip(image.geometry())

# Bare ground emissivity function for band 14.
def emiss_bare_band14(image):
    return image.expression(
        '(EM - 0.99*fvc)/(1.0 - fvc)',
        {
            'EM': aster.select('emissivity_band14').multiply(0.001),
            'fvc': aster_fvc
        }
    ).clip(image.geometry())

# Example usage:
# image = ee.Image(<your image here>)
# result_band10 = emiss_bare_band10(image)
