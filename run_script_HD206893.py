#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:11:12 2020

@author: lewis, iain
"""

from naco_pip import input_dataset, raw_dataset, calib_dataset

clas = input_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/raw/','/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/classified/', coro = True)

#clas.bad_columns()
#clas.mk_dico()
#clas.find_sky_in_sci_cube(plot = 'save')
#clas.find_derot_angles()

calib = raw_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/classified/','/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/calibrated/', final_sz = None)

#calib.dark_subtract(debug = False, plot = False)
##calib.fix_sporatic_columns(quadrant='topleft', xpixels_from_center = 7, interval = 8, verbose = True, debug = False)
#calib.flat_field_correction(debug = False, plot = False)
#calib.correct_nan(debug = False, plot = False)
#calib.correct_bad_pixels(debug = False, plot = False)
#calib.first_frames_removal(debug = False, plot = False)
#calib.get_stellar_psf(debug = False, plot = False)
#calib.subtract_sky(debug = False, plot = False)

preproc = calib_dataset('/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/calibrated/','/home/ihammond/pd87_scratch/products/NACO_archive/6_HD206893/preproc/', final_sz = 47, recenter_method = 'speckle', recenter_model = 'gauss', coro=True)

#preproc.recenter(nproc = 1, sigfactor = 4, subi_size = 21, verbose = True, debug = False, plot = False, coro = True)
preproc.bad_frame_removal(pxl_shift_thres = 0.4, crop_size = 31, verbose = True, debug = False, plot = 'save')
#preproc.crop_cube(arcsecond_diameter = 3, verbose = True, full_output= False)
#preproc.median_binning(binning_factor = 1, verbose = True)
