import ee
ee.Initialize(project='ee-hadat')

def addBand(image):
    """
    Matches atmospheric water vapour data from the NCEP reanalysis to a Landsat image.
    The function interpolates the TPW (Total Precipitable Water) values from the 6-hourly model
    times to the image acquisition time, and adds two bands: 'TPW' and 'TPWpos'.
    
    INPUT:
      image: ee.Image with a 'system:time_start' property.
    
    OUTPUT:
      ee.Image with two additional bands:
          - 'TPW': Total Precipitable Water values.
          - 'TPWpos': Position index for the SMW algorithm coefficient LUT.
    """
    # Select the day of interest from the image time
    date = ee.Date(image.get('system:time_start'))
    year = ee.Number.parse(date.format('yyyy'))
    month = ee.Number.parse(date.format('MM'))
    day = ee.Number.parse(date.format('dd'))
    date1 = ee.Date.fromYMD(year, month, day)
    date2 = date1.advance(1, 'days')
    
    # Function to compute the time difference from the Landsat image date
    def datedist(img):
        diff = ee.Number(img.get('system:time_start')).subtract(date.millis()).abs()
        return img.set('DateDist', diff)
    
    # Load the atmospheric data collection and filter by the day of interest
    TPWcollection = ee.ImageCollection('NCEP_RE/surface_wv') \
                      .filterDate(date1.format('yyyy-MM-dd'), date2.format('yyyy-MM-dd')) \
                      .map(datedist)
    
    # Select the two closest model times
    closest = TPWcollection.sort('DateDist').toList(2)
    
    # Check if there is atmospheric data for the day; if not, use a constant image with -999.0
    tpw1 = ee.Image(ee.Algorithms.If(
        closest.size().eq(0),
        ee.Image.constant(-999.0),
        ee.Image(closest.get(0)).select('pr_wtr')
    ))
    tpw2 = ee.Image(ee.Algorithms.If(
        closest.size().eq(0),
        ee.Image.constant(-999.0),
        ee.Algorithms.If(
            closest.size().eq(1),
            tpw1,
            ee.Image(closest.get(1)).select('pr_wtr')
        )
    ))
    
    # Compute time differences normalized by 21600000 (6 hours in milliseconds)
    time1 = ee.Number(ee.Algorithms.If(
        closest.size().eq(0),
        1.0,
        ee.Number(tpw1.get('DateDist')).divide(21600000)
    ))
    time2 = ee.Number(ee.Algorithms.If(
        closest.size().lt(2),
        0.0,
        ee.Number(tpw2.get('DateDist')).divide(21600000)
    ))
    
    # Interpolate TPW values
    tpw = tpw1.expression(
        'tpw1*time2 + tpw2*time1',
        {
            'tpw1': tpw1,
            'time1': time1,
            'tpw2': tpw2,
            'time2': time2
        }
    ).clip(image.geometry())
    
    # Bin TPW values to compute the lookup index for SMW coefficients
    pos = tpw.expression(
        "value = (TPW>0 && TPW<=6) ? 0" +
        ": (TPW>6 && TPW<=12) ? 1" +
        ": (TPW>12 && TPW<=18) ? 2" +
        ": (TPW>18 && TPW<=24) ? 3" +
        ": (TPW>24 && TPW<=30) ? 4" +
        ": (TPW>30 && TPW<=36) ? 5" +
        ": (TPW>36 && TPW<=42) ? 6" +
        ": (TPW>42 && TPW<=48) ? 7" +
        ": (TPW>48 && TPW<=54) ? 8" +
        ": (TPW>54) ? 9" +
        ": 0",
        {'TPW': tpw}
    ).clip(image.geometry())
    
    # Add the new bands to the image
    withTPW = image.addBands(tpw.rename('TPW')).addBands(pos.rename('TPWpos'))
    return withTPW

# Example usage:
# image = ee.Image(<your Landsat image here>)
# image_with_tpw = addBand(image)
# print(image_with_tpw.getInfo())
