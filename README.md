# nuts-for-ysos
**See Willett et al. 2024 for a full description of the methodology and the model used in nuts-for-ysos, as well as the interpretation of example results from nuts-for-ysos.**

**See main_notebook.ipynb for an example of how to use the nuts-for-ysos tool.** 

This is a Python tool for determining, via Bayesian inference, the accretion luminosities of YSOs (Young Stellar Objects) along with their effective temperatures, stellar luminosities, and extinction. The tool uses the NUTS (No U-Turn Sampler) implemented through PyMC. This project is a niche application of PyMC; if you want to learn more about PyMC in general, check out https://www.pymc.io/welcome.html. 
In the 'utility' folder is the procedure used in Willett et. al to measure emission line fluxes, though this isn't an incorporated part of nuts-for-ysos.

nuts-for-ysos was originally written to analyze spectra from the VIRUS spectrograph at the Hobby Eberly Telescope, but can be customized for other spectra-- just take note of the "input YSO spectrum requirements" section below.
Included in this repository are several example VIRUS YSO spectra from Willett et al. 2024.

In brief:
The user must provide a spectrum of an accreting YSO in the UV-Optical range. The user must also make a list of spectral features (eg. values, slopes, ratios of values, and photometric magnitudes) that will be used to fit the accreting YSO model to the data. Then, nuts-for-ysos fits the model which consists of two components:
  1. a plane-parallel slab of hydrogen in local thermodynamic equilibrium (LTE) to represent emission from accretion.
  2. a non-accreting Class III YSO template spectrum, which is linearly interpolated from a set of actual Class III YSOs of varying spectral types -- by default, the set observed with the X-Shooter spectrograph (Manara et. al 2013, 2017).
     
NUTS seeks the best-fitting model, and outputs a trace of all the parameters. Besides the model parameters themselves, traces for the stellar luminosity and accretion luminosity are also computed. The user can ultimately use this trace to derive the YSO mass and mass accretion rate using their favorite stellar evolution model (for example, Willett et. al uses Baraffe et. al 2015 and Siess et. al 2000 models)

## Packages/software used
- Python 3.12.4
- pymc 5.16.1 (see [installation instructions](https://www.pymc.io/projects/docs/en/v5.16.1/installation.html))
- included within the PyMC install:
  - arviz 0.19.0 
  - h5py 3.11.0
  - numpy 1.26.4
  - pytensor 2.23.0 
  - scipy 1.14.0
- astropy 6.1.0
- corner 2.2.2
- matplotlib 3.9.1
- for the separate emission line flux code in the 'utility' folder: specutils 1.18.0 and pyspeckit 1.0.3

## Input YSO spectrum requirements
Wavelength Range:
The wavelength array cannot extend outside the range of the Class III templates. For the default X-Shooter templates provided, the wavelengths must be above 3300.0 Angstroms and below 10189.0 Angstroms.

Resolution: 
- Users should only use this tool on spectra with wavelength arrays spaced out by $\Delta \lambda = 0.3$ Angstroms or more. The provided X-Shooter template spectra are typically defined every 0.2 Angstroms except for a few (SO797, SO641 and SO999) that are defined every 0.3 Angstorms. The input spectrum therefore shouldn't be higher resolution than these template spectra-- the code was built to make the templates lower resolution as needed to match the data, but not the other way around.
- The lower-resolution the input spectrum, the faster the PyMC sampling will generally proceed! This is because within the provided wavelength range, the model is computed with the same resolution as the data. A lower resolution means less points in wavelength space where the model has to be calculated for each and every NUTS iteration. Therefore it is advised to input lower-resolution spectra when possible (this is often all that is needed anyhow when evaluating broad features of the spectral continuum).  

Ancilliary Data: 
 - You must include the distance to the input YSO-- either as just a number, or with optional upper and lower limits (see main_notebook.ipynb for an example of this).
 - Photometric measurements for the YSO (eg. PanSTARRS gmag, rmag, and imag) can optionally be included.

## Customizable aspects of nuts-for-ysos
- You can choose the particular spectrum features (values, slopes, ratios of values, and photometric magnitudes) that are used in determining the best-fit model. See main_notebook.ipynb for more information and a demonstration.
- The grid of Class III templates can be changed. The nuts-for-ysos code uses the subset of templates listed in template_parameters_set.csv, but all available templates from Manara et al 2013 and Manara et al 2017 are separately listed in the file template_parameters_all_M13_M17.csv. Any template from template_parameters_all_M13_M17.csv can be added to template_parameters_set.csv to automatically include it in the grid. It's also possible to implement other Class III templates, such as the newer ones in Claes et al 2024. For access to these templates and their interpolable grid, see the FRAPPE github repository (https://github.com/RikClaes/FRAPPE). To include these templates in nuts-for-ysos, see the explanation in the nuts-for-ysos main_notebook.ipynb. All you'll need to do is provide the array of spectral fluxes, the wavelength range, the Teffs, and the luminosities. The only hard requirement for *any* grid of templates is that the NUTS sampler must be able to traverse the interpolable set, ie. the fluxes of the templates should vary somewhat smoothly with spectral type so the sampler doesn't get 'stuck'. The sampler will linearly interpolate between the supplied templates, their Teffs and their luminosities. See Section  4.3.1 in Willett et al for a more detailed discussion on this.
- This template-spectral-uncertainties branch is a new branch that allows the user to include uncertainties in the template spectra within the model-fitting process.
- For the reddening law contained in the model, the Rv can be changed from the default of 3.1 in both the least_squares_fit_function and pymc_NUTS_fitting function. The nuts-for-ysos code uses the Cardelli et al 1989 reddening law.
- If analyzing a VIRUS spectrum, you can use the from_example_h5file() function in VIRUS_target_prep.py to extract the spectrum for analysis. Within this function you can choose whether or not to rescale the spectrum based off the PanSTARRS g magnitude listed in the h5file, using the argument magscale=True or magscale=False.

# go nuts!
DOI: [10.5281/zenodo.13955221](https://doi.org/10.5281/zenodo.13955222)

![nuts-for-ysos](https://img.freepik.com/premium-photo/squirrel-with-outer-space-background_839169-22689.jpg)
