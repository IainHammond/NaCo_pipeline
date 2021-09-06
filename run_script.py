#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:11:12 2020
@author: iain, lewis
"""
from naco_pip import input_dataset, raw_dataset, calib_dataset, preproc_dataset

# quick basic checks
import os
from vip_hci.conf import get_available_memory

sep = '―' * 45  # used in printing functions
print('\n'+sep+'\n'+'Starting NaCo pipeline (Hammond et al. 2021)'+'\n'+sep+'\n')
try:
    nproc = int(os.getenv('SLURM_CPUS_PER_TASK',default=1))
except:
    nproc=1
get_available_memory()
print('Number of CPUs: {} \n'.format(nproc),flush=True)

# NaCo info
wavelength = 3.8e-6  # meters
size_telescope = 8.2  # meters
pixel_scale = 0.027208  # arcsecs per pixel, Launhardt et al. 2020, +/- 0.0000088

# ***************************************** PARAMETERS TO CHANGE *******************************************************

# CQTau
source = 'CQTau' # used in some saved filenames and plots, without spaces
details = '(NACO+AGPM)' # info displayed in plots and figures
ndit_sci = [100] # number of frames per science cube
ndit_sky = [100] # number of frames per sky cube
ndit_unsat = [500,400] #number of frames in unsaturated cubes
dit_sci = 0.35 #integration time for science frames
dit_unsat = 0.05 #integration time for unsaturated non coronagraphic images
dit_flat = 0.2 #integration time for flat frames
fast_reduction = False # for super fast calibration and pre-processing (median combines all science cubes into one cube)

# dictionary to pass through the pipeline saving all the static dataset information. Can be ignored
dataset_dict = {'wavelength':wavelength,'size_telescope':size_telescope,'pixel_scale':pixel_scale, 'source': source,
                'details': details,'ndit_sci': ndit_sci, 'ndit_sky':ndit_sky,'ndit_unsat':ndit_unsat,'dit_sci':dit_sci,
                'dit_unsat':dit_unsat,'dit_unsat':dit_unsat,'dit_flat':dit_flat,'fast_reduction':fast_reduction}

# ************************* Activate various functions and set inpath + outpaths ***************************************

# clas = input_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/raw/',
#                      '/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/classified/', dataset_dict,coro = True)

# clas.bad_columns()
# clas.mk_dico()
# clas.find_sky_in_sci_cube(plot = 'save')
# clas.find_derot_angles()

# calib = raw_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/classified/',
#                     '/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/calibrated/', dataset_dict,final_sz = None)

# calib.dark_subtract(bad_quadrant = [3], debug = False, plot = 'save')
# ##calib.fix_sporadic_columns(quadrant='topright', xpixels_from_center = 7, interval = 8, verbose = True, debug = False)
# calib.flat_field_correction(debug = False, plot = 'save')
# calib.correct_nan(debug = False, plot = 'save')
# calib.correct_bad_pixels(debug = False, plot = 'save')
# calib.first_frames_removal(nd_filter = False, verbose = True, debug = False, plot = 'save')
# calib.get_stellar_psf(nd_filter = False, debug = False, plot = 'save')
# calib.subtract_sky(npc = 1, debug = False, plot = 'save')

preproc = calib_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/calibrated/',
                        '/home/ihammond/pd87_scratch/products/NACO_archive/10_CQTau/preproc/', dataset_dict,
                        recenter_method = 'speckle', recenter_model = 'gauss', coro=True)

preproc.recenter(nproc=nproc, sigfactor=4, subi_size=41, crop_sz=251, verbose=True, debug=False, plot='save', coro=True)
preproc.bad_frame_removal(pxl_shift_thres=0.4, sub_frame_sz=31, verbose=True, debug=False, plot='save')
### for PCA in concentric annuli, a cropped cube is needed at minimum ###
preproc.crop_cube(arcsecond_diameter=2.5, verbose=True, debug=False)
preproc.median_binning(binning_factor=10, verbose=True)

# postproc = preproc_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/12_Elias2-24/preproc/',
#                             '/home/ihammond/pd87_scratch/products/NACO_archive/12_Elias2-24/postproc_fixbkg/',
#                            dataset_dict, nproc=nproc, npc=30)
#
# postproc.postprocessing(do_adi=False, do_adi_contrast=False, do_pca_full=False, do_pca_ann=True, cropped=True,
#                         do_snr_map=True, do_snr_map_opt=True, delta_rot=(0.5,3), mask_IWA=1, overwrite=True, plot=True,
#                         verbose=True, debug=True)
# postproc.do_negfc(do_firstguess=True, guess_xy=[(63,56)], mcmc_negfc=True, inject_neg=True, ncomp=20,
#                   algo='pca_annular', nwalkers_ini=120, niteration_min = 25, niteration_limit=10000, delta_rot=(0.5,3),
#                   weights=False, coronagraph=False, overwrite=True, save_plot=True, verbose=True)

# some previous data sets:

#MWC480
# source = 'MWC480' # used in some saved filenames and plots
# details = '(NACO+AGPM)' # info displayed in plots and figures
# ndit_sci = [100] #number of frames per science cube
# ndit_sky = [100] # number of frames per sky cube
# ndit_unsat = [400] #number of frames in unsaturated cubes
# dit_sci = 0.35 #integration time for science frames in seconds
# dit_unsat = 0.05 #integration time for unsaturated non coronagraphic images in seconds
# dit_flat = 0.2 #integration time for flat frames in seconds
# fast_reduction = False # for super fast calibration and pre-processing (median combines all science cubes into one cube)

#HD179218
#source = 'HD179218' # used in some saved filenames and plots
#details = '(NACO+AGPM)' # info displayed in plots and figures
#dit_sci = 0.35 #integration time for sci
#ndit_sci = [100] #number of frames per cube
#ndit_sky = [100] # number of frames per sky cube
#dit_unsat = 0.07 #integration time for unsaturated non coronagraphic images
#ndit_unsat = [400] #number of unsaturated frames in cubes

#HD206893
# source = 'HD206893' # used in some saved filenames and plots
# details = '(NACO+AGPM)' # info displayed in plots and figures
# ndit_sci = [200] #number of frames per science cube
# ndit_sky = [50] # number of frames per sky cube
# ndit_unsat = [100] #number of frames in unsaturated cubes
# dit_sci = 0.3 #integration time for science frames
# dit_unsat = 0.1 #integration time for unsaturated non coronagraphic images
# dit_flat = 0.2 #integration time for flat frames