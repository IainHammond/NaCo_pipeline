NaCo Pipeline - planet hunting pipeline
=======================================================

Introduction
------------

``naco_pip`` is a Python based pipeline for processing (most) VLT/NaCo + AGPM data sets from ESO, incorporating a vast array of functions from VIP Python package with optimal settings.

Capability
------------
- correction of bad columns in the lower left quadrant
- automatic classification of calibration files
- derotation angle calculation for ADI
- dark subtraction via PCA
- flat fielding (with or without airmass)
- bad/hot/NaN pixel correction
- remove sporadic DIT frames + re-scale frames
- sky subtraction via principal component analysis
- recentering of star behind AGPM (Neg 2D Gaussian or Double 2D Gaussian)
- automatic bad frame detection 
- frame cropping and binning
- median ADI and contrast curves
- PCA in full frame and PCA annular
- S/N maps
- fake companion injection and principal component optimisation
- Nelder-Mead minimisation and MCMC Sampling for source location
- NEGFC injection and post-processing

Known Issues:
------------
- occasional imperfect horizontal bar correction with PCA dark subtraction
- slow correction of bad pixels

Installation
------------
Either clone it directly, or fork it first (the latter if you plan to contribute/debug).

If you first fork the repository, click on the fork button in the top right corner.
Then clone it:

.. code-block:: bash

  $ git clone https://github.com/<replace-by-your-username>/NACO_pipeline

Or simply clone the repository to still benefit from the ``git`` syncing
functionalities:

.. code-block:: bash

  $ git clone https://github.com/IainHammond/NACO_pipeline

It is highly recommended to create a dedicated
conda environment to avoid conflicting dependencies. This can be done easily with

.. code-block:: bash

  $ conda create -n naco_pipeline python=3.9

To install ``naco_pip``, simply ``cd`` into the NACO_pipeline directory and run the setup file
in 'develop' mode:

.. code-block:: bash

  $ cd NACO_pipeline
  $ python setup.py develop

If cloned from your fork, make sure to link your NACO_pipeline directory to the upstream
source, to be able to easily update your local copy when a new version comes
out or a bug is fixed:

.. code-block:: bash

  $ git add remote upstream https://github.com/IainHammond/NACO_pipeline


Requirements (bundled)
------------
The following are installed with the pipeline:

- VIP: https://github.com/vortex-exoplanet/VIP
- hciplot: https://github.com/carlos-gg/hciplot
- numba (for bad pixel correction): https://numba.pydata.org/

Usage
------------
A Python script called ``example_run_script.py`` is bundled with the code. This file is updated often with new options and improvements. To run the code:

    1. Download a dataset from the ESO archive, including all raw calibration files.
    2. Uncompress the files and place them in a folder called "raw".
    3. Make a copy of ``example_run_script.py``
    4. Modify the ``path`` variable to point to the directory where "raw" is located
    5. Update the run script with the number of frames and integration time of the science, sky and unsaturated cubes. We recommend ``dfitspy`` (https://astrom-tom.github.io/dfitspy/build/html/index.html) for this
    6. Activate your conda environment if you made one, and start the reduction with ``python <your-run-script.py>``

Conventions
------------
- ``naco_pip`` and ``VIP`` use odd-sized frames, with the star on the central pixel.
- All systematics are corrected for during processing, including the True North offset.
- The pipeline will use half the available cores if the number of available processors is not provided in the run script. If run on a cluster with slurm, it will automatically use all processors assigned to the job.

Acknowledgements:
------------
If you use this pipeline, please cite `Hammond et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.6109H>`_. This repository is maintained by Iain Hammond (iain.hammond@monash.edu), with significant contributions by Dr Valentin Christiaens and Lewis Picker.
