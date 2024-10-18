import numpy as np

""" these functions extract various features from the spectrum. 
- The 'avg_point_feature' is an mean over some wavelength range with endpoints at left_end and right_end.
- The 'ratio_feature' is the ratio between two points (or more precisely the ratio of two averages, each over a small wavelength range).
- The 'slope_feature' is the difference between two points (or more precisely the difference of two averages, each over a small wavelength range).
- The 'photometry_feature' is the convolution of the spectrum with a photometric filter, yielding the apparent magnitude in that filter.

If the user wants to make their own custom type of feature to use in the model fitting, this is the place to include a new function for such a feature.
""" 

def avg_point_feature(wave, left_end, right_end, spectrum):
    value = np.nanmean(spectrum[(np.isclose(wave, left_end)).nonzero()[0][0]:(np.isclose(wave, right_end)).nonzero()[0][0]])
    return value

def avg_point_feature_err(wave, left_end, right_end, spectrum, err):
    value_err = 0
    avg_err_range = (err[(np.isclose(wave, left_end)).nonzero()[0][0]:(np.isclose(wave, right_end)).nonzero()[0][0]])
    for element in avg_err_range:
        if np.isnan(element) == False: 
            value_err += (element**2)
    value_err = (value_err**(1/2))/len(avg_err_range)
    return value_err

def ratio_feature(wave, left_end_num, right_end_num, left_end_denom, right_end_denom, spectrum):
    num = np.nanmean(spectrum[(np.isclose(wave, left_end_num)).nonzero()[0][0]:(np.isclose(wave, right_end_num)).nonzero()[0][0]])
    denom = np.nanmean(spectrum[(np.isclose(wave, left_end_denom)).nonzero()[0][0]:(np.isclose(wave, right_end_denom)).nonzero()[0][0]])
    ratio = num/denom
    return ratio

def ratio_feature_err(wave, left_end_num, right_end_num, left_end_denom, right_end_denom, spectrum, err):
    num = np.nanmean(spectrum[(np.isclose(wave, left_end_num)).nonzero()[0][0]:(np.isclose(wave, right_end_num)).nonzero()[0][0]])
    denom = np.nanmean(spectrum[(np.isclose(wave, left_end_denom)).nonzero()[0][0]:(np.isclose(wave, right_end_denom)).nonzero()[0][0]])
    num_err = 0
    denom_err = 0
    num_err_range = (err[(np.isclose(wave, left_end_num)).nonzero()[0][0]:(np.isclose(wave, right_end_num)).nonzero()[0][0]])
    for element in num_err_range:
        if np.isnan(element) == False: 
            num_err += (element**2)
    num_err = (num_err**(1/2))/len(num_err_range)
    denom_err_range = (err[(np.isclose(wave, left_end_denom)).nonzero()[0][0]:(np.isclose(wave, right_end_denom)).nonzero()[0][0]])
    for element in denom_err_range:
        if np.isnan(element) == False: 
            denom_err += (element**2)
    denom_err = (denom_err**(1/2))/len(denom_err_range)
    ratio = num/denom
    ratio_err_1 = (num_err / num)**2
    ratio_err_2 = (denom_err / denom)**2
    ratio_err = ratio* ((ratio_err_1 + ratio_err_2)**(1/2))
    return ratio_err

def slope_feature(wave, left_end_y1, right_end_y1, left_end_y2, right_end_y2, spectrum):
    left = np.nanmean(spectrum[(np.isclose(wave, left_end_y1)).nonzero()[0][0]:(np.isclose(wave, right_end_y1)).nonzero()[0][0]])
    right = np.nanmean(spectrum[(np.isclose(wave, left_end_y2)).nonzero()[0][0]:(np.isclose(wave, right_end_y2)).nonzero()[0][0]])
    slope = right-left
    return slope

def slope_feature_err(wave, left_end_y1, right_end_y1, left_end_y2, right_end_y2, spectrum, err):
    left_err = 0
    left_err_range = (err[(np.isclose(wave, left_end_y1)).nonzero()[0][0]:(np.isclose(wave, right_end_y1)).nonzero()[0][0]])
    for element in left_err_range:
        if np.isnan(element) == False:
            left_err += (element**2)
    left_err = (left_err**(1/2))/len(left_err_range)
    right_err = 0
    right_err_range = (err[(np.isclose(wave, left_end_y2)).nonzero()[0][0]:(np.isclose(wave, right_end_y2)).nonzero()[0][0]])
    for element in right_err_range:
        if np.isnan(element) == False:
            right_err += (element**2)
    right_err = (right_err**(1/2))/len(right_err_range)
    slope_err = ((left_err)**2 + (right_err)**2)**(1/2)
    return slope_err

def photometry_feature(wave, filt, spectrum):
    c = 2.99792458 * (1e10)
    spectrum = 1e-17*spectrum #spectrum in ergs/s/cm^2/A
    spectrum = np.ma.masked_invalid(spectrum)
    temp_spectrum = spectrum * 1e29 * (wave**2) / (c*(10**8)) #units conversion
    nu =  c / wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    mag = np.dot((nu/dnu*temp_spectrum), filt) / np.sum((nu/dnu*filt))
    Mag = -2.5 * np.log10(mag) + 23.9
    return Mag
    
def photometry_feature_err(wave, filt, spectrum, err):
    c = 2.99792458 * (1e10)
    spectrum = 1e-17*spectrum #spectrum in ergs/s/cm^2/A
    spectrum = np.ma.masked_invalid(spectrum)
    temp_spectrum = spectrum * 1e29 * (wave**2) / (c*(10**8)) #units conversion
    spectrum_err = 1e-17*err #spectrum in ergs/s/cm^2/A
    spectrum_err = np.ma.masked_invalid(spectrum_err)
    temp_spectrum_err = spectrum_err * 1e29 * (wave**2) / (c*(10**8)) #units conversion
    nu =  c / wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    mag = np.dot((nu/dnu*temp_spectrum), filt) / np.sum((nu/dnu*filt))
    mag_err = np.sqrt(np.dot(((nu/dnu)*temp_spectrum_err)**2, filt**2)) / abs(np.sum((nu/dnu*filt)))
    Mag_err = 2.5*0.434*(abs(mag_err/mag))
    return Mag_err




