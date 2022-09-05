# nuts-for-ysos
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
