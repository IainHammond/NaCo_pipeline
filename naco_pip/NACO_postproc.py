#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 27 Aug 2020 15:45:23 

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['preproc_dataset']

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, isdir
import os
import pandas
from pandas import DataFrame as DF

import vip_hci as vip
from vip_hci.fits import open_fits, write_fits
from vip_hci.pca import pca, pca_annular
from vip_hci.metrics import normalize_psf,snrmap,contrast_curve
from vip_hci.medsub import median_sub
from vip_hci.var import mask_circle,frame_filter_lowpass

class preproc_dataset:  #this class is for post-processing of the pre-processed data
    def __init__(self,inpath,outpath,dataset_dict,nproc,npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        self.fwhm = open_fits(self.inpath + 'fwhm.fits', verbose = debug)[0] # fwhm is first entry
        if verbose:
            print('FWHM:',self.fwhm)
        self.dataset_dict = dataset_dict
        self.pixel_scale = dataset_dict['pixel_scale']

    def postprocessing(self, do_adi=True, do_adi_contrast=True, do_pca_full=True, do_pca_ann=False, cropped=True,
                       do_snr_map=True, do_snr_map_opt=True, delta_rot=(0.5,3), plot=True, verbose=True, debug=False):
        """ 
        For post processing the master cube via median ADI, PCA-ADI, PCA-ann. Includes constrast curves and SNR maps.

        Parameters:
        ***********
        do_adi : bool
            Whether to do a median-ADI processing
        do_adi_contrast : bool
            Whether to compute contrast curve associated to median-ADI
        do_pca_full : bool
            Whether to apply PCA-ADI on full frame
        do_pca_ann : bool, default is False
            Whether to apply PCA-ADI in concentric annuli (more computer intensive). Only runs if cropped=True
        cropped : bool
            whether the master cube was cropped in pre-processing
        do_snr_map : bool
            whether to compute an SNR map (warning: computer intensive); useful only when point-like features are seen
            in the image
        do_snr_map_opt : bool
            Whether to compute a non-conventional (more realistic) SNR map
        delta_rot : tuple
            Threshold in rotation angle used in pca_annular to include frames in the PCA library (provided in terms of
            FWHM). See description of pca_annular() for more details
        plot : bool
            Whether to save plots to the output path (PDF file, print quality)
        verbose : bool
            prints more output when True                 
 
        """
        # ensures the correct inpath to the pre-processed data using the provided method and model
        # if self.inpath != calib.outpath + '{}_{}/'.format(recenter_method, recenter_model):
        #     self.inpath = calib.outpath + '{}_{}/'.format(recenter_method, recenter_model)
        #     print('Alert: Input path corrected. This likely occurred due to an input path typo')
        if verbose:
            print('Input path is {}'.format(self.inpath))
        details = dataset_dict['details']
        source = dataset_dict['source']
        tn_shift = -0.58 # Milli et al 2017

        ADI_cube_name = '{}_master_cube.fits'    # template name for input master cube
        derot_ang_name = 'derot_angles.fits'     # template name for corresponding input derotation angles
        psf_name = "master_unsat_psf.fits"       # name of the non-coroangraphic stellar PSF
        psfn_name = "master_unsat_psf_norm.fits" # normalised PSF
        flux_psf_name = "master_unsat-stellarpsf_fluxes.fits" # flux in a FWHM aperture found in calibration

        ADI_cube = open_fits(self.inpath+ADI_cube_name.format(source),verbose=verbose)
        derot_angles = open_fits(self.inpath+derot_ang_name,verbose=verbose)+tn_shift
        psf = open_fits(self.inpath+psf_name,verbose=verbose)
        psfn = open_fits(self.inpath+psfn_name,verbose=verbose)
        starphot = open_fits(self.inpath+flux_psf_name,verbose=verbose)[1] # scaled fwhm flux is the second entry

        mask_IWA = 1                                                        # size of numerical mask hiding the inner part of post-processed images. Provided in terms of fwhm. 1px for NACO
        mask_IWA_px = mask_IWA*self.fwhm
        if verbose:
            print("adopted mask size: {:.0f}".format(mask_IWA_px))
        
        ann_sz=3                                                             # if PCA-ADI in a single annulus or in concentric annuli, this is the size of the annulus/i in FWHM
        svd_mode = 'lapack'                                                  # python package used for Singular Value Decomposition for PCA reductions
        n_randsvd = 3                                                        # if svd package is set to 'randsvd' number of times we do PCA rand-svd, before taking the median of all results (there is a risk of significant self-subtraction when just doing it once)
        ref_cube = None                                                      # if any, load here a centered calibrated cube of reference star observations - would then be used for PCA instead of the SCI cube itself

        # Overwrite?
        overwrite_ADI = True                                                 # whether to overwrite output median-ADI files
        overwrite_pp = False                                                 # whether to overwrite output PCA-ADI files

        # TEST number of principal components
        ## PCA-FULL       
        if do_pca_full:
            test_pcs_full= list(range(1,self.npc+1))
        # PCA-ANN
        if do_pca_ann:
            test_pcs_ann = list(range(1,self.npc+1)) # needs a cropped cube

        # make directories if they don't exist
        outpath_sub = self.outpath + "sub_npc{}/".format(self.npc)

        if not isdir(self.outpath):
            os.system("mkdir "+self.outpath)
        if not isdir(outpath_sub):
            os.system("mkdir "+outpath_sub)

        ######################### Simple ADI ###########################
        if do_adi:
            if not isfile(outpath_sub+'final_ADI_simple.fits') or overwrite_ADI or debug:
                if debug: # saves the residuals
                    tmp, _, tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                             full_output=debug, verbose=verbose)
                    tmp = mask_circle(tmp,mask_IWA_px)
                    write_fits(outpath_sub+'TMP_ADI_simple_cube_der.fits', tmp,verbose=verbose)

                # make median combination of the de-rotated cube.
                tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                             radius_int=0, asize=snn_sz, delta_rot=delta_rot,
                                             full_output=False, verbose=verbose)
                tmp_tmp = mask_circle(tmp_tmp,mask_IWA_px)  # we mask the IWA
                write_fits(outpath_sub+'final_ADI_simple.fits', tmp_tmp, verbose=verbose)

            ## SNR map
            if (not isfile(outpath_sub+'final_ADI_simple_snrmap.fits') or overwrite_ADI) and do_snr_map:
                tmp = open_fits(outpath_sub+'final_ADI_simple.fits',verbose=verbose)
                tmp = mask_circle(tmp,mask_IWA_px)
                tmp_tmp = snrmap(tmp, self.fwhm,nproc=nproc, verbose=verbose)
                write_fits(outpath_sub+'final_ADI_simple_snrmap.fits', tmp_tmp, verbose=verbose)

            ## Contrast curve
            if do_adi_contrast:
                pn_contr_curve_adi = contrast_curve(ADI_cube, derot_angles, psfn, self.fwhm, pxscale=self.pixel_scale,
                                                    starphot=starphot, algo=median_sub, sigma=5., nbranch=1, theta=0,
                                                    inner_rad=1, wedge=(0,360), student=True, transmission=None,
                                                    smooth=True, plot=plot, dpi=300, debug=debug,
                                                    save_plot=outpath_sub, verbose=verbose)

        ####################### PCA-ADI full ###########################
        if do_pca_full:

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
           write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', tmp_tmp, verbose=verbose)

           if do_snr_map_opt:
               write_fits(outpath_sub + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',
                              tmp_tmp_tmp_tmp,verbose=verbose)
           ### Convolution
           if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits'):
               tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits',verbose=verbose)
               for nn in range(tmp.shape[0]):
                   tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size = self.fwhm, mode='conv')
               write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str  + '_conv.fits',
                          tmp, verbose=verbose)
           ### SNR map
           if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits')) and do_snr_map:
               tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits',verbose=verbose)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=nproc, verbose=verbose)
                   tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
               write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits',
                          tmp, verbose=verbose)
            ### SNR map optimized
           if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str +'_snrmap_opt.fits') or overwrite_pp) \
                   and do_snr_map_opt:
               tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str  + '.fits',verbose=verbose)
               tmp_tmp = open_fits(outpath_sub+ 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',verbose=verbose)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp],self.fwhm, array2=tmp_tmp_tmp_tmp[pp],incl_neg_lobes=False,
                                    nproc=nproc, verbose=verbose)
                   tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
               write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits',
                          tmp, verbose=verbose)

           ######################## PCA-ADI annular #######################
           if do_pca_ann:
               if cropped==False:
                   raise ValueError('PCA-ADI annular requires a cropped cube!')
               PCA_ADI_cube = ADI_cube.copy()
               del ADI_cube
               test_pcs_str_list = [str(x) for x in test_pcs_ann]
               ntest_pcs = len(test_pcs_ann)
               test_pcs_str = "npc" + "-".join(test_pcs_str_list)

               tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
               if do_snr_map_opt:
                   tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])

               for pp, npc in enumerate(test_pcs_ann):
                   tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                             radius_int=mask_IWA_px, fwhm=self.whm, asize=ann_sz*self.fwhm,
                                             n_segments=1, delta_rot=delta_rot, delta_sep=(0.1, 1), ncomp=int(npc),
                                             svd_mode=svd_mode, nproc=self.nproc, min_frames_lib=max(npc, 10),
                                             max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                             interpolation='lanczos4', collapse='median', ifs_collapse_range='all',
                                             full_output=False, verbose=verbose)

                   if do_snr_map_opt:
                       tmp_tmp_tmp_tmp[pp] = pca_annular(PCA_ADI_cube, -derot_angles, cube_ref=ref_cube,
                                                         scale_list=None, radius_int=mask_IWA_px, fwhm=self.fwhm,
                                                         asize=ann_sz*self.fwhm, n_segments=1, delta_rot=delta_rot,
                                                         delta_sep=(0.1, 1), ncomp=int(npc), svd_mode=svd_mode,
                                                         nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                         max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                         interpolation='lanczos4', collapse='median',
                                                         ifs_collapse_range='all', full_output=False, verbose=verbose)

               write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp, verbose=verbose)
               if do_snr_map_opt:
                   write_fits(outpath_sub + 'neg_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp_tmp_tmp,verbose=verbose)

               ### Convolution
               if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits'):
                   tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
                   for nn in range(tmp.shape[0]):
                       tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size=self.fwhm, mode='conv')
                   write_fits(outpath_sub+'final_PCA-ADI_ann_'+test_pcs_str+'_conv.fits',
                              tmp, verbose=verbose)

               ### SNR map
               if (not isfile(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits')) and do_snr_map:
                   tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
                   for pp in range(ntest_pcs):
                       tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=nproc, verbose=verbose)
                       tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                   write_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits', tmp, verbose=verbose)
               ### SNR map optimized
               if (not isfile(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits'))and do_snr_map_opt:
                   tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits', verbose=verbose)
                   tmp_tmp = open_fits(outpath_sub +'neg_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
                   for pp in range(ntest_pcs):
                       tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=plot, array2=tmp_tmp_tmp_tmp[pp], nproc=nproc,
                                        verbose=verbose)
                       tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                   write_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits',tmp, verbose=verbose)
        
                
