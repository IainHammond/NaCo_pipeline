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
from vip_hci.metrics import normalize_psf
from naco_pip import fits_info

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
    

class postproc_dataset():  #this class is for post-processing of the pre-processed data
    def __init__(self,inpath,outpath,nproc,npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        self.fwhm = open_fits(self.inpath + 'fwhm.fits')[0] # fwhm is first entry
        if verbose:
            print('FWHM:',self.fwhm)
        self.plsc_ori = fits_info.pixel_scale       


    def pca(self, recenter_method, recenter_model, cropped = True, binning_factor, do_adi = True, do_pca_full = True, do_pca_ann = False, do_snr_map = True, do_snr_map_opt = True, delta_rot = (0.5,3), verbose = True, debug = False):        
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
            Whether to apply PCA-ADI in concentric annuli (more computer intensive)
        
        do_snr_map : bool
            whether to compute an SNR map (warning: computer intensive); useful only when point-like features are seen in the image
            
        do_snr_map_opt : bool
            Whether to compute a non-conventional (more realistic) SNR map 
            
        delta_rot : tuple
            Threshold in rotation angle used in pca_annular to include frames in the PCA library (provided in terms of FWHM). See description of pca_annular() for more details
        
        verbose : bool
            prints more output when True                 
 
        """
        details = fits_info.details
        source = fits_info.source
        ADI_cube_name = 'master_cube_{}_{}_good_frames_cropped_bin{}.fits'                 # template name for input master cubes (i.e. the recentered bad-frame trimmed ones)
        derot_ang_name = 'derot_angles_{}_{}_good_frames_bin{}.fits'                       # template name for corresponding input derotation angles 
        psf_name = "master_unsat_psf.fits"                                                 # name of the non-coroangraphic stellar PSF
        psfn_name = "master_unsat_psf_norm.fits" 
        mask_IWA = 0.5
        mask_IWA_px = int(mask_IWA*self.fwhm)
        print("adopted mask size: {:.0f}".format(mask_IWA_px))
        tn_shift = -0.58                                                                      # size of numerical mask hiding the inner part of post-processed images. Provided in terms of fwhm.
        outpath_sub = self.outpath+"Sub_bin{}_{}_{}/"
        
        ann_sz=3                                                                       # if PCA-ADI in a single annulus or in concentric annuli, this is the size of the annulus/i in FWHM
        svd_mode_sann = 'lapack'                                                       # python package used for Singular Value Decomposition in PCA-ADI sann
        svd_mode_1 = 'lapack'                                                          # python package used for Singular Value Decomposition for PCA reductions on uncropped cube
        svd_mode_2 = 'lapack'                                                          # python package used for Singular Value Decomposition for PCA reductions on cropped cube (Note: 'lapack' is more robust but 'randsvd' is significantly faster)        
        svd_mode_all = [svd_mode_1,svd_mode_2]                                         
        n_randsvd = 3                                                                  # if svd package is set to 'randsvd' number of times we do PCA rand-svd, before taking the median of all results (there is a risk of significant self-subtraction when just doing it once)
        ref_cube = None                                                                # if any, load here a centered calibrated cube of reference star observations - would then be used for PCA instead of the SCI cube itself 

        # Overwrite?
        overwrite_ADI = True                                                           # whether to overwrite output median-ADI files
        overwrite_pp = False         
        
        # TEST number of principal components - for cropped and uncropped cubes
        ## PCA-FULL       
        test_pcs_full_crop = list(range(1,16))
        test_pcs_full_nocrop = list(range(1,16))
        test_pcs_full_all = [test_pcs_full_crop,test_pcs_full_nocrop]
        # PCA-ANN
        test_pcs_ann_crop = list(range(1,16))
        test_pcs_ann_nocrop = list(range(1,16))
        test_pcs_ann_all = [test_pcs_ann_crop, test_pcs_ann_nocrop]
        
        if not isdir(self.outpath):
            os.system("mkdir "+self.outpath)
        if not isdir(outpath_sub.format(binning_factor,recenter_method, recenter_model)):
            os.system("mkdir "+outpath_sub.format(binning_factor,recenter_method, recenter_model))            
        
        oversamp_fac = 1
        plsc = plsc_ori/oversamp_fac
        
        ADI_cube = open_fits(self.inpath+ADI_cube_name.format(recenter_method, recenter_model,binning_factor))
        derot_angles = open_fits(self.inpath+derot_ang_name.format(recenter_method, recenter_model,binning_factor))+tn_shift
        psf = open_fits(self.inpath+psf_name)
        psfn, starphot, fwhm = normalize_psf(psf, fwhm='fit', size=None, threshold=None, mask_core=None,
                                             model='gauss', imlib='opencv', interpolation='lanczos4',
                                             force_odd=True, full_output=True, verbose=verbose, debug=debug)        
        
        if do_adi:
            if not isfile(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits') or overwrite_ADI or debug:
                if debug:
                    tmp, _, tmp_tmp = vip.medsub.median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot, 
                                             full_output=debug, verbose=verbose)
                    tmp = vip.var.mask_circle(tmp,0.9*self.fwhm)
                    write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'TMP_ADI_simple_cube_der.fits', tmp)
                else:
                    tmp_tmp = vip.medsub.median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot, 
                                             full_output=debug, verbose=verbose)
                tmp_tmp = vip.var.mask_circle(tmp_tmp,mask_IWA*self.fwhm)  # we mask the IWA
                write_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits', tmp_tmp)
            else:
                tmp_tmp = open_fits(outpath_sub.format(binning_factor,recenter_method,recenter_model)+'final_ADI_simple.fits')        
        
        
        
                
