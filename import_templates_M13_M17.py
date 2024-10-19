from scipy import interpolate
from scipy.optimize import curve_fit
import numpy as np
import math
from astropy.table import Table 
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

def make_def_wave_model(def_wave_data, def_wave_templates):
    #makes a wavelength array with the same spacing as the data, but the range of the templates
    def_wave_data_trimmed = def_wave_data[(def_wave_data>=np.min(def_wave_templates))*(def_wave_data<=np.max(def_wave_templates))] #trim data wavelength range to be within that of the templates
    spacing_left = float(np.diff(def_wave_data_trimmed)[0])
    spacing_right = float(np.diff(def_wave_data_trimmed)[-1])
    if def_wave_data_trimmed[0]>np.min(def_wave_templates)+spacing_left:
        def_wave_ext_left = np.arange(np.min(def_wave_templates), def_wave_data_trimmed[0], spacing_left)
    else:
        def_wave_ext_left = np.array([])
    if def_wave_data_trimmed[-1]+spacing_right < np.max(def_wave_templates):
        def_wave_ext_right = np.arange(def_wave_data_trimmed[-1]+spacing_right, np.max(def_wave_templates), spacing_right)
    else:
        def_wave_ext_right = np.array([])
    def_wave_model = np.concatenate((def_wave_ext_left,def_wave_data_trimmed,def_wave_ext_right))
    return def_wave_model

def reduce_resolution(def_wave_model, def_wave_templates, flux_templates, mean_resolution):
    """for making templates match the lower resolution of the user-inputted spectrum
    def_wave_model : numpy array
        the desired wavelength range for the templates (lower resolution)
    def_wave_templates : numpy array
       the original wavelength array for the templates
    flux_templates : numpy array
        the original flux array for the templates
    mean_resolution : float
        the desired resolution of the templates
    """
    if np.min(np.diff(def_wave_model)) < np.min(np.diff(def_wave_templates)):
        print('Warning: inputted data wavelength array contains a spacing of less than the templates. Data with a higher resolution than the templates should first be converted to a resolution equal to or lower than the templates.')
    def gauss(w, sigma, mu):
        var = sigma**2
        exp = (-((w-mu)**2)/(2*var)) 
        N = ((var*2*math.pi)**(-1/2)) * np.exp(exp)
        return N
    FWHM = np.mean(def_wave_model)/mean_resolution
    sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = Gaussian1DKernel(stddev = sigma)
    convolved_flux_template_total = convolve(flux_templates, gauss, boundary='extend')
    interp_template_total = np.interp(def_wave_model, def_wave_templates, convolved_flux_template_total, left=0.0, right=0.0)
    return interp_template_total

def prep_scale_templates_M13_M17(def_wave_model, mean_resolution):

    """ Generates the set of class III templates used for model fitting, with the correct resolution to match the resolution of the target spectrum. The function rescales the raw templates (taken from Manara+2013, Manara+2017) so that their median flux values follow a polynomial with temperature Teff. This makes it easier for the NUTS sampler to eventually interpolate between the templates later on.
You can adjust which templates from Manara+ 2013, 2017 are included/excluded from the grid, by altering the file template_parameters_set.csv
    
    Parameters
    ----------
    def_wave_data : numpy array
        The array of wavelength values covered by the target spectrum in Angstroms.
    mean_resolution : int, float
        The mean resolution of the spectrum over the wavelength range.

    Returns
    -------
    template_Teffs : numpy array
        The effective temperature Teff associated with each template in Kelvin.
    template_lums_new : numpy array
        The new luminosity of the templates after they have been rescaled, in logarithmic solar units log(L/L_sun).
    def_wave_model : numpy array
        The wavelength array for the template spectra in Angstroms.
    templates_scaled : numpy array
        The flux values for the template spectra, in units of 10^-17 erg/s/cm2/Angstrom.
    """
    
    #for M dwarfs beyond M0 the SpT-->Teff scale is from Luhman et al. 2003: https://ui.adsabs.harvard.edu/abs/2003ApJ...593.1093L/abstract
    #for M0, as well as G and K stars: table A5 of Kenyon and Hartmann 1995: https://ui.adsabs.harvard.edu/abs/1995ApJS..101..117K/abstract
    M_SpTs = [0,1,2,3,4,5,6,7,8,9,9.5]
    M_Teffs = [3850,3705, 3560, 3415, 3270, 3125, 2990, 2880, 2710, 2400, 2330]
    M_scale = interpolate.interp1d(M_SpTs, M_Teffs)
    K_SpTs = [0,1,2,3,4,5,6,7]
    K_Teffs = [5250, 5080, 4900, 4730, 4590, 4350, 4205, 4060]
    K_scale = interpolate.interp1d(K_SpTs, K_Teffs)
    G_SpTs = [5,6,7,8,9]
    G_Teffs = [5770, 5700, 5630, 5520, 5410]
    G_scale = interpolate.interp1d(G_SpTs, G_Teffs)

    def get_temperature(SpT):
        number = float(SpT[1:])
        if SpT[0] == 'M':
            Teff = M_scale(number)
        if SpT[0] == 'K':
            Teff = K_scale(number)
        if SpT[0] == 'G':
            Teff = G_scale(number)
        return Teff

    def rescale_function(x, a, c, d):
        y = a*((x-c)**4) +d
        return y

    #load in templates
    template_table = Table.read('template_parameters_set.csv')
    templates_dist_scaled = []
    templates_scaled = []
    template_names = []
    template_SpTs = []
    template_lums = []
    template_Teffs = []
    medians = []

    for t in range(0, len(template_table)):
        template_name = template_table['Name'][t]
        d_template = template_table['Dist'][t]
        SpT = template_table['SpT'][t]
        temperature = float(get_temperature(SpT))
        L_template_log = template_table['log_L__'][t]

        #UVB portion
        if template_table['source'][t]=='Manara+2017':
            file_UVB = open('Class_III_templates/UVB_templates/Manara_2017/'+template_name+'.txt', "r")
        elif template_table['source'][t]=='Manara+2013':
            file_UVB = open('Class_III_templates/UVB_templates/Manara_2013/'+template_name+'.txt', "r")
        rawtxt = file_UVB.read()
        wave_template_UVB = []
        flux_template_UVB = []
        temp = (rawtxt.split('\n')[1:])
        for row in temp:
            splitted = (row[1:].split(' '))
            if len(splitted)>1:
                splitted2 = []
                for s in splitted:
                    if s!= '':
                        splitted2.append(s)
                wave_template_UVB.append(float(splitted2[0]))
                flux_template_UVB.append(float(splitted2[1]))
        wave_template_UVB = np.array(wave_template_UVB)*10    #angstroms
        flux_template_UVB = np.array(flux_template_UVB)/10    #ergs/s/cm^2/A

        #VIS portion
        if template_table['source'][t]=='Manara+2017':
            file_VIS = open('Class_III_templates/VIS_templates/Manara_2017/'+template_name+'.txt', "r")
        elif template_table['source'][t]=='Manara+2013':
            file_VIS = open('Class_III_templates/VIS_templates/Manara_2013/'+template_name+'.txt', "r")
        rawtxt = file_VIS.read()
        wave_template_VIS = []
        flux_template_VIS = []
        temp = (rawtxt.split('\n')[1:])
        for row in temp:
            splitted = (row[1:].split(' '))
            if len(splitted)>1:
                splitted2 = []
                for s in splitted:
                    if s!= '':
                        splitted2.append(s)
                wave_template_VIS.append(float(splitted2[0]))
                flux_template_VIS.append(float(splitted2[1]))
        wave_template_VIS = np.array(wave_template_VIS)*10    #angstroms
        flux_template_VIS = np.array(flux_template_VIS)/10    #ergs/s/cm^2/A

        wave_template_total = np.concatenate((wave_template_UVB, wave_template_VIS))
        flux_template_total = np.concatenate((flux_template_UVB, flux_template_VIS))
        interp_template_total = reduce_resolution(def_wave_model, wave_template_total, flux_template_total, mean_resolution)

        og_photosphere = interp_template_total* 1e17 # units of 10^-17 ergs/s/cm^2/A
        dist_scaled_photosphere = og_photosphere*(d_template**2)

        templates_dist_scaled.append(dist_scaled_photosphere)
        template_Teffs.append(temperature)
        template_SpTs.append(SpT)
        template_lums.append(L_template_log)
        template_names.append(template_name)
        medians.append(np.median(dist_scaled_photosphere[int(np.where(np.abs(def_wave_model-4500)==np.min(np.abs(def_wave_model-4500)))[0][0]):int(np.where(np.abs(def_wave_model-5500)==np.min(np.abs(def_wave_model-5500)))[0][0])]))

    template_Teffs = np.array(template_Teffs)
    template_lums = np.array(template_lums)

    template_lums_new = []
    for t in range(0, len(template_Teffs)):
        SpT = template_SpTs[t]
        scaling_new = rescale_function(template_Teffs[t], 6.7125630196248415e-06, 2400, 1300)
        factor = (scaling_new/ medians[t])
        template_scaled = templates_dist_scaled[t]*factor
        templates_scaled.append(template_scaled)
        L_template_log_new = np.log10((10**(template_lums[t]))*factor)
        template_lums_new.append(L_template_log_new)

    return [template_Teffs, template_lums_new, templates_scaled]




