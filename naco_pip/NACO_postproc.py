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

    def pca(self, recenter_method, recenter_model, do_adi, do_pca_full, do_pca_ann, verbose = True, debug = False):
             
        ######################### Simple ADI ###########################
        if do_adi:
            if not isfile(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'.fits') or overwrite_ADI or debug:
                if debug:
                    tmp, _, tmp_tmp = vip.medsub.median_sub(ADI_cube, derot_angles, fwhm=fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot, 
                                             full_output=debug, verbose=True)
                    tmp = vip.var.mask_circle(tmp,0.9*fwhm)
                    write_fits(outpath_sub.format(bin_fac,cen_met)+'TMP_ADI_simple_cube_der'+label+label_ov+label_test+'.fits', tmp)
                else:
                    tmp_tmp = vip.medsub.median_sub(ADI_cube, derot_angles, fwhm=fwhm,
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot, 
                                             full_output=debug, verbose=True)
                tmp_tmp = vip.var.mask_circle(tmp_tmp,mask_IWA*fwhm)  # we mask the IWA
                write_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'.fits', tmp_tmp)
            else:
                tmp_tmp = open_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'.fits')
            #id_snr_adi_df[counter] = vip.metrics.snr(tmp_tmp, (xx_comp,yy_comp), fwhm)
            ## Convolution
            if (not isfile(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'_conv.fits') or overwrite_ADI) and do_conv:
                tmp = open_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'.fits')
                tmp = vip.var.frame_filter_gaussian2d(tmp, fwhm/2, mode='conv')
                write_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'_conv.fits', tmp, verbose=False)
            ## SNR map  
            if (not isfile(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'_snrmap.fits') or overwrite_ADI) and do_snr_map:
                tmp = open_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'.fits')
                rad_in = mask_IWA 
                tmp = vip.var.mask_circle(tmp,rad_in*fwhm)
                tmp_tmp = vip.metrics.snrmap(tmp, fwhm)
                write_fits(outpath_sub.format(bin_fac,cen_met)+'final_ADI_simple'+label+label_ov+label_test+'_snrmap.fits', tmp_tmp, verbose=False)
            ## Contrast curve ADI
            if do_adi_contrast:
                #psfn = open_fits(outpath_3+'master_unsat_psf_norm'+label_ov+'.fits')
                #starphot = open_fits(inpath+'fluxes.fits')[ii]
                pn_contr_curve_adi = vip.metrics.contrast_curve(ADI_cube, derot_angles, psfn,
                                                             fwhm, plsc, starphot=starphot, 
                                                             algo=vip.madi.adi, sigma=5., nbranch=1, 
                                                             theta=0, inner_rad=1, wedge=(0,360),
                                                             student=True, transmission=None, smooth=True,
                                                             plot=False, dpi=100, debug=False, 
                                                             verbose=verbose)
                arr_dist_adi = np.array(pn_contr_curve_adi['distance'])
                arr_contrast_adi = np.array(pn_contr_curve_adi['sensitivity (Student)'])
                for ff in range(nfcp):
                    idx = find_nearest(arr_dist_adi, rad_arr[ff])
                    sensitivity_5sig_adi_df[ff] = arr_contrast_adi[idx]
            ADI_cube = None 


