from astropy.table import Table
import numpy as np

def from_example_h5file(h5filename, h5index, magscale=False):
    #for example:
    #h5filename = '20190101_0000021.h5'
    #h5index=52

    """Takes a provided .h5 file containing data from a VIRUS observation, and extracts a particular spectrum from the .h5 file.
    It also cleans the spectrum, removing segments which have a low relative weight. 
    It can scale the spectrum to match the object's Pan-STARRS g magnitude.

    Parameters
    ----------
    h5filename : str
        The path of the .h5 file.
    h5index : int
        The index of the desired spectrum.
    magscale : bool, optional
        Whether or not to scale the spectrum to match the object's Pan-STARRS g magnitude (default is False).

    Returns
    -------
    YSO : numpy array
        An array of the flux values for the YSO spectrum, in units of 10^-17 erg/s/cm2/Angstrom.
    YSO_err : numpy array
        An array of associated errors in the flux valus, in units of 10^-17 erg/s/cm2/Angstrom.
    """

    #magscale is whether or not to rescale flux based off PanSTARRS g mag
    import h5py

    def_wave_UVB = np.arange(3470, 5542, 2)
    c = 2.99792458 * (1e10)

    f = h5py.File(h5filename, 'r')

    T = Table.read('VIRUS_normalization.txt', format='ascii.fixed_width_two_line')
    flux_normalization = np.array(T['normalization'])
    spectrum = f['CatSpectra']['spectrum'][h5index]*1e-17*flux_normalization #erg/s/cm2
    weight = f['CatSpectra']['weight'][h5index]
    error = f['CatSpectra']['error'][h5index]*1e-17*flux_normalization
    ps_gmag_target = f['CatSpectra']['gmag'][h5index]
    
    ps2g_file = Table.read('psg_filter_curve.csv')
    ps2g_wave=ps2g_file ['wavelength (A)']
    ps2g_val=ps2g_file ['transmission']
    ps2g_wave = np.array(ps2g_wave)
    ps2g_val = np.array(ps2g_val)
    filtg = np.interp(def_wave_UVB, ps2g_wave, ps2g_val, left=0.0, right=0.0)
    
    temp_target_spectrum = spectrum * 1e29 * (def_wave_UVB**2) / (c*(10**8))
    temp_target_spectrum_err = error * 1e29 * (def_wave_UVB**2) / (c*(10**8))
    nu =  c / def_wave_UVB
    dnu = np.diff(nu)
    dnu = np.hstack([dnu[0], dnu])
    #low weights removal (weight being less than 0.5x the median weight) and we'll interpolate over them
    gmask = np.isfinite(temp_target_spectrum) * (weight > (0.5*np.nanmedian(weight)))
    target_gmag = np.dot((nu/dnu*temp_target_spectrum)[gmask], filtg[gmask]) / np.sum((nu/dnu*filtg)[gmask])
    target_gmag_err = np.dot((nu/dnu*temp_target_spectrum_err)[gmask], filtg[gmask]) / np.sum((nu/dnu*filtg)[gmask])
    target_GMag = -2.5 * np.log10(target_gmag) + 23.9
    target_GMag_err = -2.5 * 0.434*(target_gmag_err/target_gmag)

    #PanSTARRS scaling of data
    if magscale == True:
        magscaling_target = 10**(-0.4*(ps_gmag_target - target_GMag))
        magscaling_target_err = abs(magscaling_target*2.303*(-0.4*target_GMag_err))
        YSO_pieces = (spectrum*magscaling_target)[gmask]
        YSO_err_pieces = YSO_pieces*((((error/spectrum)**2 + (magscaling_target_err/magscaling_target)**2)**(1/2))[gmask])
    
    elif magscale == False:
        YSO_pieces = (spectrum)[gmask]
        YSO_err_pieces = YSO_pieces*((((error/spectrum)**2)**(1/2))[gmask])
        
    #final YSO spectrum
    YSO = np.interp(def_wave_UVB, def_wave_UVB[gmask], YSO_pieces, left=0.0, right=0.0) * 1e17
    YSO_err = np.interp(def_wave_UVB, def_wave_UVB[gmask], YSO_err_pieces, left=0.0, right=0.0)* 1e17

    return YSO, YSO_err
