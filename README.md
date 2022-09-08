# nuts-for-ysos
**See [insert paper] for a full description of the methodology and the model used in nuts-for-ysos, as well as the interpretation of example results from nuts-for-ysos.** 

This is a python tool for determining, via Bayesian inference, the accretion luminosity of YSOs (Young Stellar Objects) in conjunction with their effective temperature, stellar luminosity, and extinction. The tool uses the NUTS (No U-Turn Sampler) implemented through PyMC3. 

nuts-for-ysos was written by default for spectra from the VIRUS spectrograph (cite) but can be customized for other spectra; take note of the "input YSO spectrum requirements" section below.

In brief:
The user must provide a spectrum of an accreting YSO in the UV-Optical range. Then, nuts-for-ysos fits a model to the YSO consisting of two components: (INCLUDE CITATIONS)
  1. a plane-parallel slab of hydrogen in local thermodynamic equilibrium (LTE) to represent emission from accretion.
  2. a non-accreting Class III YSO template spectrum
NUTS seeks the best-fitting model, and outputs a trace of all the parameters. Besides the model parameters themselves, the trace for the stellar luminosity and accretion luminosity are also computed. The user can ultimately use the trace to derive a probability distribution of the YSO's mass accretion rate (see [insert paper] for details)

## package requirements (WIP)

## input YSO spectrum requirements (WIP)
The spectrum can be low or high resolution

## customizable aspects of nuts-for-ysos (WIP)

## useful links (WIP)

# go nuts!

To-do list:
- Finish up the notebook that puts everything together and shows how to interpret results
- Write useful annotations in notebook
- Make sure each py file isn't importing anything unnecessary
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
