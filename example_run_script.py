#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Iain
"""
from naco_pip import input_dataset, raw_dataset, calib_dataset, preproc_dataset
from os import getenv
from multiprocessing import cpu_count
from vip_hci.config import get_available_memory

sep = '―' * 45  # used in printing functions
print('\n'+sep+'\n'+'Starting NaCo pipeline (Hammond et al. 2022)'+'\n'+sep+'\n')
try:
    nproc = int(getenv('SLURM_CPUS_PER_TASK', default=1)) * int(getenv('SLURM_NNODES'))
except:
    nproc = int(cpu_count()/2)  # set to cpu_count()/2 for efficiency
get_available_memory()
print('Using {} CPUs\n'.format(nproc), flush=True)

# VLT/NaCo info, doesn't need to be changed
wavelength = 3.78e-6  # meters
size_telescope = 8.2  # meters
pixel_scale = 0.027208  # arcsecs per pixel, Launhardt et al. 2020, +/- 0.0000088

# ***************************************** PARAMETERS TO CHANGE *******************************************************

# example HD206893
source = 'HD206893'       # used in some saved filenames and plots
details = '(NACO+AGPM)'   # info displayed in plots and figures
ndit_sci = [200]          # number of frames per science cube
ndit_sky = [50]           # number of frames per sky cube
ndit_unsat = [100]        # number of frames in unsaturated cubes
dit_sci = 0.3             # integration time for science frames
dit_unsat = 0.1           # integration time for unsaturated non coronagraphic images
dit_flat = 0.2            # integration time for flat frames
fast_calibration = False  # uses the median frame only to identify bad pixels, only use if really needed

# Can be ignored. Dictionary to pass through the pipeline saving all the static dataset information.
dataset_dict = {'wavelength': wavelength, 'size_telescope': size_telescope, 'pixel_scale': pixel_scale,
                'source': source, 'details': details, 'ndit_sci': ndit_sci, 'ndit_sky':ndit_sky,
                'ndit_unsat':ndit_unsat, 'dit_sci': dit_sci,  'dit_unsat': dit_unsat, 'dit_flat':dit_flat,
                'fast_calibration': fast_calibration, 'nproc': nproc}

# ************************* Activate various functions and set common path ***************************************
path = '/your/common/path/to/data/'

clas = input_dataset(inpath=path + 'raw/', outpath=path + 'classified/', dataset_dict=dataset_dict, coro=True)
clas.bad_columns(correct=True, overwrite=False, sat_val=32768, plot=True, verbose=True, debug=False)
clas.mk_dico(plot=True, verbose=True, debug=False)
clas.find_sky_in_sci_cube(nres=3, coro=True, plot=True, verbose=True, debug=False)
clas.find_derot_angles(plot=True, verbose=True, debug=False)

calib = raw_dataset(inpath=path+'classified/', outpath=path+'calibrated/', dataset_dict=dataset_dict, final_sz=None)
calib.dark_subtract(method='median', bad_quadrant=[3], plot=True, verbose=True, debug=False)
calib.flat_field_correction(plot=True, debug=False)
calib.correct_nan(debug=False)
calib.correct_bad_pixels(verbose=True, plot=True, overwrite=False, debug=False)
calib.first_frames_removal(nrm='auto', verbose=True, plot=True, debug=False)
calib.get_stellar_psf(nd_filter=False, plot=True, verbose=True, debug=False)
calib.subtract_sky(mode='pca', npc=1, debug=False, plot=True)

preproc = calib_dataset(inpath=path+'calibrated/', outpath=path+'preproc/', dataset_dict=dataset_dict,
                        recenter_method='speckle', recenter_model='gauss', coro=True)
preproc.recenter(sigfactor=4, subi_size=41, crop_sz=251, verbose=True, debug=False, plot=True, coro=True)
preproc.bad_frame_removal(pxl_shift_thres=0.4, sub_frame_sz=31, verbose=True, debug=False, plot=True)
preproc.crop_cube(arcsecond_diameter=3, verbose=True, debug=False)  # required for PCA-ADI annular and contrast curves
preproc.median_binning(binning_factor=1, verbose=True)  # speeds up PCA-ADI annular and contrast curves, reduces S/N

postproc = preproc_dataset(inpath=path+'preproc/', outpath=path+'postproc/', dataset_dict=dataset_dict, nproc=nproc, npc=20) # npc can be int (for a single number of Principal
                                                                                                                             # Components), tuple or list for a range and step
postproc.postprocessing(do_adi=True, do_adi_contrast=True, do_pca_full=True, do_pca_ann=True, fake_planet=False,
                        first_guess_skip=True, fcp_pos=[0.3], firstguess_pcs=[1, 5, 1], do_snr_map=True,
                        do_snr_map_opt=True, planet_pos=None, delta_rot=(0.5, 3), mask_IWA=1, coronagraph=True,
                        overwrite=True, verbose=True, debug=False)
# only to be run if there is a source/blob detected by running post-processing
# postproc.do_negfc(do_firstguess=True, guess_xy=[(63,56)], mcmc_negfc=True, inject_neg=True, ncomp=20,
#                   algo='pca_annular', nwalkers_ini=120, niteration_min = 25, niteration_limit=10000, delta_rot=(0.5,3),
#                   weights=False, coronagraph=False, overwrite=True, save_plot=True, verbose=True)
