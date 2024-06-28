# nuts-for-ysos
**See Willett et. al (in prep) for a full description of the methodology and the model used in nuts-for-ysos, as well as the interpretation of example results from nuts-for-ysos.** 

**See main_notebook.ipynb for an example of how to use the nuts-for-ysos tool.** 

This is a Python tool for determining, via Bayesian inference, the accretion luminosities of YSOs (Young Stellar Objects) along with their effective temperatures, stellar luminosities, and extinction. The tool uses the NUTS (No U-Turn Sampler) implemented through PyMC. This project is a niche application of PyMC; if you want to learn more about PyMC in general, check out https://www.pymc.io/welcome.html. 

nuts-for-ysos was originally written to analyze spectra from the VIRUS spectrograph at the Hobby Eberly Telescope, but can be customized for other spectra-- just take note of the "input YSO spectrum requirements" section below.
Included in this repository are several example VIRUS YSO spectra from Willett et. al (in prep).

In brief:
The user must provide a spectrum of an accreting YSO in the UV-Optical range. Then, nuts-for-ysos fits a model to the YSO consisting of two components:
  1. a plane-parallel slab of hydrogen in local thermodynamic equilibrium (LTE) to represent emission from accretion.
  2. a non-accreting Class III YSO template spectrum, observed with the X-Shooter spectrograph (Manara et. al 2013, 2017).
     
NUTS seeks the best-fitting model, and outputs a trace of all the parameters. Besides the model parameters themselves, traces for the stellar luminosity and accretion luminosity are also computed. The user can ultimately use this trace to derive the YSO mass and mass accretion rate using their favorite stellar evolution model (for example, Willett et. al (in prep) uses Baraffe et. al 2015 and Siess et. al 2000 models)

## Packages/software used (to be updated):
- Python 3.9.16
- arviz 0.14.0
- astropy 5.2
- corner 2.2.1
- h5py 3.8.0
- matplotlib 3.6.2
- numpy 1.24.1
- pymc 5.0.1
- pytensor 2.8.11
- scipy 1.9.3

## Input YSO spectrum requirements
Resolution: 
  Users should only use this tool on spectra with wavelength arrays spaced out by $\Delta \lambda = 0.3$ Angstroms or more.
  The provided X-Shooter template spectra are typically defined every 0.2 Angstroms except for a few (SO797, SO641 and SO999) that are defined every 0.3 Angstorms. The input spectrum shouldn't be higher resolution than these template spectra-- the code was built to make the templates lower resolution as needed, but not the other way around.

Wavelength Range:
The wavelength array cannot extend below 3300.0 Angstroms or above 10189.0 Angstroms.

Ancilliary Data: 
 - You must include the distance to the input YSO-- either as just a number, or with optional upper and lower limits.
 - Photometric measurements (eg. PanSTARRS gmag, rmag, and imag) are optional.

## Customizable aspects of nuts-for-ysos
- If analyzing a VIRUS spectrum, you can use the from_example_h5file() function in VIRUS_target_prep.py to extract the spectrum for analysis. Within this function you can choose whether or not to rescale the spectrum based off the PanSTARRS gmag listed in the h5file, using the argument magscale=True.
- You can choose the particular spectrum features (values, slopes, ratios of values, and photometric values) that are used in determining the best-fit model. See main_notebook.ipynb for more information and a demonstration.
- For the reddening law contained in the model, the Rv can be changed from the default of 3.1 in both the least squares and PyMC functions if needed. The nuts-for-ysos code uses the Cardelli et al 1989 reddening law.

# go nuts!
