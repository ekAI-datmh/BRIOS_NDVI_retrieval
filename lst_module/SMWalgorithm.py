import ee
from lst_module.SMW_coefficients import *
ee.Initialize(project='ee-hadat')

# Assume you have imported or defined SMWcoef with coefficient collections.
# For example:
# from SMW_coefficients import coeff_SMW_L4, coeff_SMW_L5, coeff_SMW_L7, coeff_SMW_L8, coeff_SMW_L9
# And then group them into an object/dictionary:
class SMWcoef:
    coeff_SMW_L4 = COEFF_SMW_L4
    coeff_SMW_L5 = COEFF_SMW_L5
    coeff_SMW_L7 = COEFF_SMW_L7
    coeff_SMW_L8 = COEFF_SMW_L8
    coeff_SMW_L9 = COEFF_SMW_L9

def get_lookup_table(fc, prop_1, prop_2):
    """
    Creates a lookup table (as an ee.List) between two properties in a FeatureCollection.
    The output is a list with two elements: the first is the list of keys and the second is the list of values.
    """
    reducer = ee.Reducer.toList().repeat(2)
    lookup = fc.reduceColumns(reducer, [prop_1, prop_2])
    return ee.List(lookup.get('list'))

def addBand(landsat):
    """
    Returns a function that, when applied to an image, computes the LST (Land Surface Temperature)
    using the Statistical Mono-Window algorithm and adds it as a new band called 'LST'.
    
    INPUTS:
       - landsat: string indicating the Landsat satellite ('L4', 'L5', 'L7', 'L8', or 'L9')
       - image: an ee.Image with the necessary bands (including 'EM', 'TPW', and a TIR band)
    
    USAGE:
       lst_func = addBand('L8')
       image_with_lst = lst_func(image)
    """
    def wrap(image):
        # Select algorithm coefficients based on the Landsat satellite.
        if landsat == 'L4':
            coeff_SMW = SMWcoef.coeff_SMW_L4
        elif landsat == 'L5':
            coeff_SMW = SMWcoef.coeff_SMW_L5
        elif landsat == 'L7':
            coeff_SMW = SMWcoef.coeff_SMW_L7
        elif landsat == 'L8':
            coeff_SMW = SMWcoef.coeff_SMW_L8
        else:
            coeff_SMW = SMWcoef.coeff_SMW_L9

        # Create lookup tables for coefficients A, B, and C.
        A_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'A')
        B_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'B')
        C_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'C')

        # Remap the image's TPW bin ('TPWpos') to the corresponding coefficient values.
        A_img = image.remap(ee.List(A_lookup.get(0)), ee.List(A_lookup.get(1)), 0.0, 'TPWpos').resample('bilinear')
        B_img = image.remap(ee.List(B_lookup.get(0)), ee.List(B_lookup.get(1)), 0.0, 'TPWpos').resample('bilinear')
        C_img = image.remap(ee.List(C_lookup.get(0)), ee.List(C_lookup.get(1)), 0.0, 'TPWpos').resample('bilinear')

        # Select the TIR band.
        # For Landsat L9 and L8, use band 'B10'; for L7 use 'B6_VCID_1'; otherwise use 'B6'
        if landsat in ['L9', 'L8']:
            tir = 'B10'
        elif landsat == 'L7':
            tir = 'B6_VCID_1'
        else:
            tir = 'B6'

        # Compute the LST using the Statistical Mono-Window algorithm.
        lst = image.expression(
            'A * Tb1 / em1 + B / em1 + C',
            {
                'A': A_img,
                'B': B_img,
                'C': C_img,
                'em1': image.select('EM'),
                'Tb1': image.select(tir)
            }
        ).updateMask(image.select('TPW').lt(0).Not())

        # Add the LST band to the original image.
        return image.addBands(lst.rename('LST'))
    return wrap

# Example usage:
# lst_func = addBand('L8')
# image_with_lst = lst_func(ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20170606'))
