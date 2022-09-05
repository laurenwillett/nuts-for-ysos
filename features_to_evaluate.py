from astropy.table import Table
import numpy as np


def avg_value_feature(wave, left_end, right_end, spectrum):
    value = np.mean(spectrum[int(np.where(wave == float(left_end))[0][0]):int(np.where(wave == float(right_end))[0][0])])
    return value

def avg_value_feature_err(wave, left_end, right_end, spectrum, err):
    value_err = 0
    avg_err_range = (err[int(np.where(wave == float(left_end))[0][0]):int(np.where(wave == float(right_end))[0][0])])
    for element in avg_err_range:
        value_err += (element**2)
    value_err = (value_err**(1/2))/len(avg_err_range)
    return value_err

def ratio_feature(wave, left_end_num, right_end_num, left_end_denom, right_end_denom, spectrum):
    num = np.mean(spectrum[int(np.where(wave == float(left_end_num))[0][0]):int(np.where(wave == float(right_end_num))[0][0])])
    denom = np.mean(spectrum[int(np.where(wave == float(left_end_denom))[0][0]):int(np.where(wave == float(right_end_denom))[0][0])])
    ratio = num/denom
    return ratio

def ratio_feature_err(wave, left_end_num, right_end_num, left_end_denom, right_end_denom, spectrum, err):
    num = np.mean(spectrum[int(np.where(wave == float(left_end_num))[0][0]):int(np.where(wave == float(right_end_num))[0][0])])
    denom = np.mean(spectrum[int(np.where(wave == float(left_end_denom))[0][0]):int(np.where(wave == float(right_end_denom))[0][0])])
    num_err = 0
    denom_err = 0
    num_err_range = (err[int(np.where(wave == float(left_end_num))[0][0]):int(np.where(wave == float(right_end_num))[0][0])])
    for element in num_err_range:
        num_err += (element**2)
    num_err = (num_err**(1/2))/len(num_err_range)
    denom_err_range = (err[int(np.where(wave == float(left_end_denom))[0][0]):int(np.where(wave == float(right_end_denom))[0][0])])
    for element in denom_err_range:
        denom_err += (element**2)
    denom_err = (denom_err**(1/2))/len(denom_err_range)
    ratio = num/denom
    ratio_err_1 = (num_err / num)**2
    ratio_err_2 = (denom_err / denom)**2
    ratio_err = ratio* ((ratio_err_1 + ratio_err_2)**(1/2))
    return ratio_err

def slope_feature(wave, left_end_y1, right_end_y1, left_end_y2, right_end_y2, spectrum):
    left = np.mean(spectrum[int(np.where(wave == float(left_end_y1))[0][0]):int(np.where(wave == float(right_end_y1))[0][0])])
    right = np.mean(spectrum[int(np.where(wave == float(left_end_y2))[0][0]):int(np.where(wave == float(right_end_y2))[0][0])])
    slope = right-left
    return slope

def slope_feature_err(wave, left_end_y1, right_end_y1, left_end_y2, right_end_y2, spectrum, err):
    left_err = 0
    left_err_range = (err[int(np.where(wave == float(left_end_y1))[0][0]):int(np.where(wave == float(right_end_y1))[0][0])])
    for element in left_err_range:
        left_err += (element**2)
    left_err = (left_err**(1/2))/len(left_err_range)
    right_err = 0
    right_err_range = (err[int(np.where(wave == float(left_end_y2))[0][0]):int(np.where(wave == float(right_end_y2))[0][0])])
    for element in right_err_range:
        right_err += (element**2)
    right_err = (right_err**(1/2))/len(right_err_range)
    slope_err = ((left_err)**2 + (right_err)**2)**(1/2)
    return slope_err

def photometry_feature(wave, spectrum, filt):
    spectrum = 1e-17*spectrum #spectrum in ergs/s/cm^2/A
    temp_spectrum = spectrum * 1e29 * (wave**2) / (c*(10**8)) #units conversion
    nu =  c / wave
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    mag = np.dot((nu/dnu*temp_spectrum), filt) / np.sum((nu/dnu*filt))
    Mag = -2.5 * np.log10(mag) + 23.9
    return Mag


def make_feature_list(wave, spectrum, err):
    #customizable
    balmer_slope = slope_feature(wave, 3504.0, 3524.0, 3580.0, 3600.0, spectrum)
    balmer_val = avg_value_feature(wave, 3520.0, 3580.0, spectrum)
    balmer_val_2 = avg_value_feature(wave, 3850.0, 3870.0, spectrum)
    purple_val = avg_value_feature(wave, 4000.0, 4030.0, spectrum)
    paschen_slope = slope_feature(wave, 3980.0, 4020.0, 4770.0, 4820.0, spectrum)
    paschen_val = avg_value_feature(wave, 4590.0, 4624.0, spectrum)
    optical_val = avg_value_feature(wave, 5090.0, 5130.0, spectrum)
    optical_val_2 = avg_value_feature(wave, 5460.0, 5490.0, spectrum)
    optical_slope_2 = slope_feature(wave, 5060.0, 5100.0, 5390.0, 5424.0, spectrum)
    features = [balmer_slope, balmer_val, balmer_val_2, purple_val, paschen_slope, paschen_val, optical_val, optical_val_2, optical_slope_2]

    if type(err) == bool:
        if err== False:
            return features
    
    else:
        balmer_slope_err = slope_feature_err(wave, 3504.0, 3524.0, 3580.0, 3600.0, spectrum, err)
        balmer_val_err = avg_value_feature_err(wave, 3520.0, 3580.0, spectrum, err)
        balmer_val_2_err = avg_value_feature_err(wave, 3850.0, 3870.0, spectrum, err)
        purple_val_err = avg_value_feature_err(wave, 4000.0, 4030.0, spectrum, err)
        paschen_slope_err = slope_feature_err(wave, 3980.0, 4020.0, 4770.0, 4820.0, spectrum, err)
        paschen_val_err = avg_value_feature_err(wave, 4590.0, 4624.0, spectrum, err)
        optical_val_err = avg_value_feature_err(wave, 5090.0, 5130.0, spectrum, err)
        optical_val_2_err = avg_value_feature_err(wave, 5460.0, 5490.0, spectrum, err)
        optical_slope_2_err = slope_feature_err(wave, 5060.0, 5100.0, 5390.0, 5424.0, spectrum, err)
        feature_errs = [balmer_slope_err, balmer_val_err, balmer_val_2_err, purple_val_err, paschen_slope_err, paschen_val_err, optical_val_err, optical_val_2_err, optical_slope_2_err]
        return features, feature_errs
    




