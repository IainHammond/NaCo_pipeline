#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 27 Aug 2020 15:45:23 

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['postproc_dataset']

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, isdir
import os
import pandas
from pandas import DataFrame as DF
#import pdb # uncomment if you want to use python debugger

import vip_hci as vip
from vip_hci.fits import open_fits, write_fits
from vip_hci.pca import pca, pca_annular
from vip_hci.metrics import normalize_psf, snrmap, contrast_curve
from vip_hci.medsub import median_sub
from vip.var import mask_circle,frame_filter_gaussian2d
#from naco_pip import fits_info

# Definition of Useful function
def find_nearest(array, value, output='index', constraint=None):
    """
    Function to find the index, and optionally the value, of an array's closest 
    element to a certain value.
    Possible outputs: 'index','value','both' 
    Possible constraints: 'ceil', 'floor', None 
        "ceil" will return the closest element with a value greater than 'value' 
        "floor" does the opposite
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")
        
    idx = (np.absolute(array-value)).argmin()
    if type == 'ceil' and array[idx]-value < 0:
        idx+=1
    elif type == 'floor' and value-array[idx] < 0:
        idx-=1

    if output=='index': return idx
    elif output=='value': return array[idx]
    else: return array[idx], idx

class postproc_dataset:  #this class is for post-processing of the pre-processed data
    def __init__(self,inpath,outpath,nproc,npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        self.fwhm = open_fits(self.inpath + 'fwhm.fits', verbose = debug)[0] # fwhm is first entry
        if verbose:
            print('FWHM:',self.fwhm)
        self.plsc_ori = fits_info.pixel_scale ########### update

    def postproc(self, recenter_method, recenter_model, binning_factor, delta_rot = (0.5,3), cropped = True, do_adi = True, do_pca_full = True, do_pca_ann = False, do_snr_map = True, do_snr_map_opt = True,  verbose = True, debug = False):
        """ 
        For post processing the master cube using PCA-ADI        

        Parameters:
        ***********
        recenter_method : str
            method used to recenter in preproc
        recenter_model : str
            model used to recenter in preproc
        cropped : bool
            whether to use cropped frames. Must be used for running PCA in concentric annuli
        binning_factor : int
            for use in the name of the master cube. will not be considered if cropped = False
        do_adi : bool
            Whether to do a median-ADI processing
        do_pca_full : bool
            Whether to apply PCA-ADI on full frame
        do_pca_ann : bool, default is False
            Whether to apply PCA-ADI in concentric annuli (more computer intensive). Only runs if cropped = True
        do_snr_map : bool
            whether to compute an SNR map (warning: computer intensive); useful only when point-like features are seen in the image
        do_snr_map_opt : bool
            Whether to compute a non-conventional (more realistic) SNR map
        delta_rot : tuple
            Threshold in rotation angle used in pca_annular to include frames in the PCA library (provided in terms of
            FWHM). See description of pca_annular() for more details
        verbose : bool
            prints more output when True                 
 
        """
        # ensures the correct inpath to the pre-processed data using the provided method and model
        if self.inpath != (calib.outpath + '{}_{}/'.format(recenter_method, recenter_model)):
            self.inpath = calib.outpath + '{}_{}/'.format(recenter_method, recenter_model)
        if debug:
            print('Input path is {}'.format(self.inpath))
        details = fits_info.details
        source = fits_info.source
        tn_shift = -0.58
        if cropped:
            ADI_cube_name = 'master_cube_good_frames_cropped_bin{}.fits'    # template name for input master cubes (i.e. the recentered bad-frame trimmed ones)
            derot_ang_name = 'derot_angles_good_frames_bin{}.fits'          # template name for corresponding input derotation angles
            ADI_cube = open_fits(self.inpath+ADI_cube_name.format(binning_factor))
            derot_angles = open_fits(self.inpath+derot_ang_name.format(binning_factor))+tn_shift
        else:
            ADI_cube_name = 'master_cube_good_frames.fits'                  # template name for input master cubes (i.e. the recentered bad-frame trimmed ones)
            derot_ang_name = 'derot_angles_good_frames.fits'                # template name for corresponding input derotation angles
            ADI_cube = open_fits(self.inpath+ADI_cube_name)
            derot_angles = open_fits(self.inpath+derot_ang_name)+tn_shift
        psf_name = "master_unsat_psf.fits"                                  # name of the non-coroangraphic stellar PSF
        psfn_name = "master_unsat_psf_norm.fits"
        psf = open_fits(self.inpath+psf_name)
        psfn = open_fits(self.inpath+psfn_name)
        mask_IWA = 1                                                      # size of numerical mask hiding the inner part of post-processed images. Provided in terms of fwhm
        mask_IWA_px = int(mask_IWA*self.fwhm)
        print("adopted mask size: {:.0f}".format(mask_IWA_px))
        
        ann_sz=3                                                             # if PCA-ADI in a single annulus or in concentric annuli, this is the size of the annulus/i in FWHM
        svd_mode = 'lapack'                                                  # python package used for Singular Value Decomposition for PCA reductions
        n_randsvd = 3                                                        # if svd package is set to 'randsvd' number of times we do PCA rand-svd, before taking the median of all results (there is a risk of significant self-subtraction when just doing it once)
        ref_cube = None                                                      # if any, load here a centered calibrated cube of reference star observations - would then be used for PCA instead of the SCI cube itself

        # Overwrite?
        overwrite_ADI = True                                                 # whether to overwrite output median-ADI files
        overwrite_pp = False                                                 # whether to overwrite output PCA-ADI files

        # TEST number of principal components - for cropped and uncropped cubes
        ## PCA-FULL       
        if do_pca_full:
            test_pcs_full_crop = list(range(1,self.npc+1))
            test_pcs_full_nocrop = list(range(1,self.npc+1))
        # PCA-ANN
        if do_pca_ann:
            test_pcs_ann = list(range(1,self.npc+1)) # pca ann will only run with a cropped cube
        
        if not isdir(self.outpath):
            os.system("mkdir "+self.outpath)
        if cropped:
            outpath_sub = self.outpath + "Sub_bin{}_{}_{}/"
            if not isdir(outpath_sub.format(binning_factor,recenter_method, recenter_model)):
                os.system("mkdir "+outpath_sub.format(binning_factor,recenter_method, recenter_model))
        else:
            outpath_sub = self.outpath + "Sub_{}_{}/"
            if not isdir(outpath_sub.format(recenter_method, recenter_model)):
                os.system("mkdir "+outpath_sub.format(recenter_method, recenter_model))
        
        oversamp_fac = 1
        plsc = plsc_ori/oversamp_fac

        # starphot is used in computing contrast curves. psfna and fwhm are already known from the calibration module

        # _, starphot, _ = normalize_psf(psf, fwhm=self.fwhm, size=None, threshold=None, mask_core=None,
        #                                      model='gauss', imlib='opencv', interpolation='lanczos4',
        #                                      force_odd=True, full_output=True, verbose=verbose, debug=debug)

        ######################### Simple ADI ###########################
        if do_adi:
            if cropped:
                if not isfile(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits') or overwrite_ADI or debug:
                    if debug:
                        tmp, _, tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                                 radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                                 full_output=debug, verbose=verbose)
                        tmp = mask_circle(tmp,mask_IWA*self.fwhm)
                        write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'TMP_ADI_simple_cube_der.fits', tmp)
                    else:
                        tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                                 radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                                 full_output=debug, verbose=verbose)
                    tmp_tmp = mask_circle(tmp_tmp,mask_IWA*self.fwhm)  # we mask the IWA
                    write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits', tmp_tmp)
                else:
                    tmp_tmp = open_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits')

                ## SNR map
                if (not isfile(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple_snrmap.fits') or overwrite_ADI) and do_snr_map:
                    tmp = open_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits')
                    rad_in = mask_IWA
                    tmp = mask_circle(tmp,rad_in*self.fwhm)
                    tmp_tmp = snrmap(tmp, self.fwhm)
                    write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple_snrmap.fits', tmp_tmp, verbose=verbose)
            if not cropped:
                if not isfile(outpath_sub.format(recenter_method,recenter_model) + 'final_ADI_simple.fits') or overwrite_ADI or debug:
                    if debug:
                        tmp, _, tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                                     radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                                     full_output=debug, verbose=verbose)
                        tmp = mask_circle(tmp, rad_in * self.fwhm)
                        write_fits(outpath_sub.format(recenter_method,recenter_model) + 'TMP_ADI_simple_cube_der.fits', tmp)
                    else:
                        tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                             full_output=debug, verbose=verbose)
                    tmp_tmp = mask_circle(tmp_tmp, mask_IWA * self.fwhm)  # we mask the IWA
                    write_fits(outpath_sub.format(recenter_method, recenter_model) + 'final_ADI_simple.fits',
                        tmp_tmp)
                else:
                    tmp_tmp = open_fits(outpath_sub.format(recenter_method, recenter_model) + 'final_ADI_simple.fits')

                    ## SNR map
                if (not isfile(outpath_sub.format(recenter_method,recenter_model) + 'final_ADI_simple_snrmap.fits') or overwrite_ADI) and do_snr_map:
                    tmp = open_fits(outpath_sub.format(recenter_method, recenter_model) + 'final_ADI_simple.fits')
                    rad_in = mask_IWA
                    tmp = mask_circle(tmp, rad_in * self.fwhm)
                    tmp_tmp = snrmap(tmp, self.fwhm)
                    write_fits(outpath_sub.format(recenter_method,recenter_model) + 'final_ADI_simple_snrmap.fits', tmp_tmp,
                               verbose=verbose)

        ####################### PCA-ADI full ###########################
        if do_pca_full:
           if crop:
               test_pcs_full = test_pcs_full_crop
           else:
               test_pcs_full = test_pcs_full_nocrop

           test_pcs_str_list = [str(x) for x in test_pcs_full]
           ntest_pcs = len(test_pcs_full)
           test_pcs_str = "npc" + "-".join(test_pcs_str_list)
           PCA_ADI_cube = ADI_cube.copy()
           tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])

           if do_snr_map_opt:
               tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
           for pp, npc in enumerate(test_pcs_full):
               if svd_mode == 'randsvd':
                   tmp_tmp_tmp = np.zeros([n_randsvd, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                   for nr in range(n_randsvd):
                       tmp_tmp_tmp[nr] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube,
                                             scale_list=None, ncomp=int(npc),
                                             svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                             delta_rot=delta_rot, fwhm=self.fwhm, collapse='median', check_memory=True,
                                             full_output=False, verbose=verbose)

                   tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
               else:
                   tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube,
                                     scale_list=None, ncomp=int(npc),
                                     svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                     delta_rot=delta_rot, fwhm=self.fwhm, collapse='median', check_memory=True,
                                     full_output=False, verbose=verbose)
                   if do_snr_map_opt:
                       tmp_tmp_tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=-derot_angles, cube_ref=ref_cube,
                                                 scale_list=None, ncomp=int(npc),
                                                 svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                                 delta_rot=delta_rot, fwhm=self.fwhm, collapse='median', check_memory=True,
                                                 full_output=False, verbose=verbose)
           if cropped:
               write_fits(outpath_sub.format(binning_factor, recenter_method,recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', tmp_tmp)
           else:
               write_fits(outpath_sub.format(recenter_method,recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', tmp_tmp)
           if do_snr_map_opt:
               if cropped:
                   write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model) + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',
                              tmp_tmp_tmp_tmp)
               else:
                   write_fits(outpath_sub.format(recenter_method,recenter_model) + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',
                              tmp_tmp_tmp_tmp)
           ### Convolution
           if not isfile(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits'):
               tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '.fits')
               for nn in range(tmp.shape[0]):
                   tmp[nn] = frame_filter_gaussian2d(tmp[nn], self.fwhm, mode='conv')
               write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str  + '_conv.fits',
                          tmp, verbose=False)
           ### SNR map
           if (not isfile(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits')) and do_snr_map:
               tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '.fits')
               rad_in = mask_IWA  # we comment it for a better visualization of the snr map (there are spurious values in the center)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False)
                   tmp[pp] = mask_circle(tmp[pp], rad_in * self.fwhm)
               write_fits(outpath_sub.format(bin_fac,
                                             cen_met) + 'final_PCA-ADI_full_' + test_pcs_str + label + label_ov + label_test + '_snrmap.fits',
                          tmp, verbose=False)
               ### SNR map  optimized
           if (not isfile(outpath_sub.format(bin_fac,
                                             cen_met) + 'final_PCA-ADI_full_' + test_pcs_str + label + label_ov + label_test + '_snrmap_opt.fits') or overwrite_pp) and do_snr_map_opt:
               tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str  + '.fits')
               tmp_tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits')
               rad_in = mask_IWA  # we comment it for a better visualization of the snr map (there are spurious values in the center)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp],self.fwhm, plot=False, array2=tmp_tmp_tmp_tmp[pp],incl_neg_lobes=False)
                   tmp[pp] =mask_circle(tmp[pp], rad_in * self.fwhm)
               write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits',
                          tmp, verbose=False)

           ######################## PCA-ADI annular #######################
           if do_pca_ann:
               if not cropped:
                   raise ValueError("PCA in concentric annuli must be performed on a cropped cube!")
               else:
                   test_pcs_str_list = [str(x) for x in test_pcs_ann]
                   ntest_pcs = len(test_pcs_ann)
                   test_pcs_str = "npc" + "-".join(test_pcs_str_list)

                   tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                   if do_snr_map_opt:
                       tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])

                   for pp, npc in enumerate(test_pcs_ann):
                       tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                                 radius_int=mask_IWA_px,
                                                 fwhm=self.whm, asize=ann_sz * self.fwhm, n_segments=1, delta_rot=delta_rot,
                                                 delta_sep=(0.1, 1), ncomp=int(npc), svd_mode=svd_mode_2, nproc=self.nproc,
                                                 min_frames_lib=max(npc, 10), max_frames_lib=200, tol=1e-1,
                                                 scaling=None,
                                                 imlib='opencv', interpolation='lanczos4', collapse='median',
                                                 ifs_collapse_range='all', full_output=False, verbose=verbose)
                       if do_snr_map_opt:
                           tmp_tmp_tmp_tmp[pp] = pca_annular(PCA_ADI_cube, -derot_angles, cube_ref=ref_cube,
                                                             scale_list=None, radius_int=mask_IWA_px,
                                                             fwhm=self.fwhm, asize=ann_sz * self.fwhm, n_segments=1,
                                                             delta_rot=delta_rot,
                                                             delta_sep=(0.1, 1), ncomp=int(npc), svd_mode=svd_mode_2,
                                                             nproc=self.nproc,
                                                             min_frames_lib=max(npc, 10), max_frames_lib=200, tol=1e-1,
                                                             scaling=None,
                                                             imlib='opencv', interpolation='lanczos4',
                                                             collapse='median',
                                                             ifs_collapse_range='all', full_output=False,
                                                             verbose=verbose)

                   write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'.fits',tmp_tmp)
                   if do_snr_map_opt:
                       write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'neg_PCA-ADI_ann_'+test_pcs_str+'.fits', tmp_tmp_tmp_tmp)
                   ### Convolution
                   if not isfile(outpath_sub.format(binning_factor, recenter_method, recenter_model) + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits'):
                       tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'.fits')
                       for nn in range(tmp.shape[0]):
                           tmp[nn] = frame_filter_gaussian2d(tmp[nn], self.fwhm, mode='conv')
                       write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'_conv.fits',
                                  tmp, verbose=False)

                   ### SNR map
                   if (not isfile(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits')) and do_snr_map:
                       tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'.fits')
                       rad_in = mask_IWA  # we comment it for a better visualization of the snr map (there are spurious values in the center)
                       for pp in range(ntest_pcs):
                           tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False)
                           tmp[pp] = mask_circle(tmp[pp], rad_in * self.fwhm)
                       write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits',
                                  tmp, verbose=False)
                   ### SNR map  optimized
                   if (not isfile(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits'))and do_snr_map_opt:
                       tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'.fits')
                       tmp_tmp = open_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'neg_PCA-ADI_ann_'+test_pcs_str+'.fits')
                       rad_in = mask_IWA  # we comment it for a better visualization of the snr map (there are spurious values in the center)
                       for pp in range(ntest_pcs):
                           tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, array2=tmp_tmp_tmp_tmp[pp])
                           tmp[pp] = mask_circle(tmp[pp], rad_in * self.fwhm)
                       write_fits(outpath_sub.format(binning_factor, recenter_method, recenter_model)+'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits',tmp, verbose=False)
        
                
