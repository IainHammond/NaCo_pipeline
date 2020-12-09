NACO Pipeline - planet detecting pipeline
=======================================================

Introduction
------------

``naco_pip`` is a Python based pipeline for processing any VLT/NaCo + AGPM data set from ESO, incorporating a vast array of functions from VIP Python package with optimal settings. 
Only the run script needs to be modified with the information of your target system, some file information and file directories

Capability
------------
- correction of bad columns in the lower left quadrant
- automatic classified of cubes and reading observation time
- derotation angle calculation for ADI
- dark subtraction via PCA
- flat fielding (with+without airmass)
- bad/hot/NaN pixel correction
- remove sporadic DIT frames + re-scale frames
- sky subtraction via PCA
- recentring of star behind AGPM (Neg 2D Gaussian or Double 2D Gaussian)
- automatic bad frame detection 
- cropping
- binning 


Under testing
------------
- median ADI and PCA in full frame
- S/N map
- fast reduction


Known Issues:
------------
- Fast reduction is not working
- plot scales aren't ideal
- sporadic columns fix may not work on all data sets

