NaCo Pipeline - planet detecting pipeline
=======================================================

Introduction
------------

``naco_pip`` is a Python based pipeline for processing (most) VLT/NaCo + AGPM data sets from ESO, incorporating a vast array of functions from VIP Python package with optimal settings. 
Only the run script needs to be modified with the information of your target system, some file information and file directories

Capability
------------
- correction of bad columns in the lower left quadrant
- automatic classified of cubes and reading observation time
- derotation angle calculation for ADI
- dark subtraction via PCA
- flat fielding (with or without airmass)
- bad/hot/NaN pixel correction
- remove sporadic DIT frames + re-scale frames
- sky subtraction via PCA
- recentring of star behind AGPM (Neg 2D Gaussian or Double 2D Gaussian)
- automatic bad frame detection 
- cropping
- binning 
- median ADI and contrast curve
- PCA in full frame and PCA annular
- S/N map
- fake companion injection and principle component optimisation (new)
- Nelder-Mead minimisation and MCMC Sampling for source location
- NEGFC injection and post-processing

Under testing
------------
- optimised contrast curve for full frame and annular PCA-ADI
- fast reduction

Known Issues:
------------
- fast reduction doesnt work in some sections
- some calibration plots are messy
- debug mode needs further testing in calibration
- in some data sets the systematic horizontal bars weren't quite removed