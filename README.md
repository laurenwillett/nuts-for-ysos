# nuts-for-ysos
**See [insert paper] for a full description of the methodology and the model used in nuts-for-ysos, as well as the interpretation of example results from nuts-for-ysos.** 

This is a python tool for determining, via Bayesian inference, the accretion luminosity of YSOs (Young Stellar Objects) in conjunction with their effective temperature, stellar luminosity, and extinction. The tool uses the NUTS (No U-Turn Sampler) implemented through PyMC3. 

nuts-for-ysos was written by default for spectra from the VIRUS spectrograph at the Hobby Eberly Telescope, but can be customized for other spectra-- just take note of the "input YSO spectrum requirements" section below.

In brief:
The user must provide a spectrum of an accreting YSO in the UV-Optical range. Then, nuts-for-ysos fits a model to the YSO consisting of two components:
  1. a plane-parallel slab of hydrogen in local thermodynamic equilibrium (LTE) to represent emission from accretion.
  2. a non-accreting Class III YSO template spectrum observed with the X-Shooter spectrograph (Manara+ 2013, 2017)
NUTS seeks the best-fitting model, and outputs a trace of all the parameters. Besides the model parameters themselves, the trace for the stellar luminosity and accretion luminosity are also computed. The user can ultimately use the trace to derive the YSO's mass accretion rate using their favorite stellar evolution model (see [insert paper] for details)

## package requirements (WIP)
- numpy 1.18.1
- theano 1.0.5
- pymc3 3.9.3
- scipy 1.5.3
- astropy 4.2.1
- corner 2.2.1
- matplotlib 3.2.2
- h5py 3.3.0

## input YSO spectrum requirements (WIP)
Resolution: 
  Users should therefore only use nuts-for-ysos on spectra with wavelength arrays spaced out by 0.3 Angstroms or more.
  The provided X-Shooter template spectra are typically defined every 0.2 Angstroms except for a few (SO797, SO641 and SO999) that are defined every 0.3 Angstorms. The input spectrum shouldn't be higher resolution than the template spectra-- the code was designed to make the templates lower resolution as needed, but not the other way around.

Wavelength Range:
The wavelength array cannot extend below 3300.0 Angstroms or above 10189.0 Angstroms.

Ancilliary Data: You need to know the distance to the input YSO, but photometric measurements (eg. PanSTARRS gmag, rmag, and imag) are not required.

## customizable aspects of nuts-for-ysos (WIP)
- If using VIRUS parallel spectra, you can use the from_example_h5file() function and choose whether or not to rescale the spectrum based off its PanSTARRS gmag.
- features_to_evaluate.py contains the particular spectrum features that are used in determining the best-fit model. These features can be altered from the default provided, just make sure you update the make_feature_list() function at the bottom of features_to_evaluate.py.
- Even with the default set of features, you can choose whether or not to evaluate PanSTARRS rmag and imag as a part of the model fitting process.
- WIP: mention how to use different photometry
- If you have errorbars on the distance to the input YSO, those can be accounted for by the NUTS sampler-- but if not, the distance can be passed as just a scalar instead.
- The Rv can be changed from the default of 3.1 . The code uses the Cardelli et al 1989 reddening law.

## useful links (WIP)

# go nuts!

To-do list:
-TEST EVERYTHING AGAIN!!
- update package versions
- Finish up the notebook that puts everything together and shows how to interpret results
- I could make import_templates.py do less unit conversions
- Write useful annotations in notebook
- Write notes in README for users about:
  - What packages to install, and their versions
  - the limitations in wavelength range for this tool
  - the resolution of the templates, the convolution to match resolution of the target YSO
  - how someone would change what features the fitting process evaluates
  - toggling whether or not to use PanSTARRS rmag and imag as a feature
  - how to switch out what photometry you use altogether
  - note the extinction law from Cardelli et al 1989 and where you'd go to change it
  - the function I made to interpret h5 files
    - toggling whether or not spectrum should be scaled to match a certain gmag (as was done with VIRUS data, but usually wouldnt be)
  - link to pymc3 information, NUTS information
