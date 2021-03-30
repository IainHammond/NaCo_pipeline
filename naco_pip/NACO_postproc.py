#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applys post-processing algorithms to the pre-processed cube

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['preproc_dataset']

import numpy as np
from os.path import isfile, isdir
import os

from vip_hci.fits import open_fits, write_fits
from vip_hci.pca import pca, pca_annular, pca_annulus
from vip_hci.metrics import snrmap,contrast_curve
from vip_hci.medsub import median_sub
from vip_hci.var import mask_circle,frame_filter_lowpass, frame_center
from vip_hci.negfc import mcmc_negfc_sampling, firstguess

class preproc_dataset:  #this class is for post-processing of the pre-processed data
    def __init__(self,inpath,outpath,dataset_dict,nproc,npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        try:
            self.fwhm = open_fits(self.inpath + 'fwhm.fits', verbose=False)[0] # fwhm is first entry
        except:
            print("Alert: No FWHM file found. Setting to median value of 4.2")
            self.fwhm = 4.2
        self.dataset_dict = dataset_dict
        self.pixel_scale = dataset_dict['pixel_scale']

    def postprocessing(self, do_adi=True, do_adi_contrast=True, do_pca_full=True, do_pca_ann=True, cropped=True,
                       do_snr_map=True, do_snr_map_opt=True, delta_rot=(0.5,3), mask_IWA=1, plot=True, verbose=True, debug=False):
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
            Whether to apply annular PCA-ADI (more computer intensive). Only runs if cropped=True
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
        mask_IWA : int, default 1
            Size of the numerical mask that hides the inner part of post-processed images. Provided in terms of fwhm
        plot : bool
            Whether to save plots to the output path (PDF file, print quality)
        verbose : bool
            prints more output when True                 
        debug : bool, default is False
            Saves extra output files
        """
        # ensures the correct inpath to the pre-processed data using the provided method and model
        # if self.inpath != calib.outpath + '{}_{}/'.format(recenter_method, recenter_model):
        #     self.inpath = calib.outpath + '{}_{}/'.format(recenter_method, recenter_model)
        #     print('Alert: Input path corrected. This likely occurred due to an input path typo')

        # make directories if they don't exist
        print("======= Starting post-processing....=======")
        outpath_sub = self.outpath + "sub_npc{}/".format(self.npc)

        if not isdir(self.outpath):
            os.system("mkdir " + self.outpath)
        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath))
            print('Output path is {}'.format(outpath_sub))

        details = self.dataset_dict['details']
        source = self.dataset_dict['source']
        tn_shift = 0.58 # Milli et al. 2017, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'    # template name for input master cube
        derot_ang_name = 'derot_angles.fits'     # template name for corresponding input derotation angles
        ADI_cube = open_fits(self.inpath+ADI_cube_name.format(source),verbose=verbose)
        derot_angles = open_fits(self.inpath+derot_ang_name,verbose=verbose)+tn_shift

        if do_adi_contrast:
            psfn_name = "master_unsat_psf_norm.fits" # normalised PSF
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits" # flux in a FWHM aperture found in calibration
            psfn = open_fits(self.inpath+psfn_name,verbose=verbose)
            starphot = open_fits(self.inpath+flux_psf_name,verbose=verbose)[1] # scaled fwhm flux is the second entry

        mask_IWA = 1                                                        # size of numerical mask hiding the inner part of post-processed images. Provided in terms of fwhm
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
                                             radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                             full_output=False, verbose=verbose)
                tmp_tmp = mask_circle(tmp_tmp,mask_IWA_px)  # we mask the IWA
                write_fits(outpath_sub+'final_ADI_simple.fits', tmp_tmp, verbose=verbose)

            ## SNR map
            if (not isfile(outpath_sub+'final_ADI_simple_snrmap.fits') or overwrite_ADI) and do_snr_map:
                tmp = open_fits(outpath_sub+'final_ADI_simple.fits',verbose=verbose)
                tmp = mask_circle(tmp,mask_IWA_px)
                tmp_tmp = snrmap(tmp, self.fwhm,nproc=self.nproc, verbose=verbose)
                write_fits(outpath_sub+'final_ADI_simple_snrmap.fits', tmp_tmp, verbose=verbose)

            ## Contrast curve
            if do_adi_contrast:
                pn_contr_curve_adi = contrast_curve(ADI_cube, derot_angles, psfn, self.fwhm, pxscale=self.pixel_scale,
                                                    starphot=starphot, algo=median_sub, sigma=5., nbranch=1, theta=0,
                                                    inner_rad=1, wedge=(0,360), student=True, transmission=None,
                                                    smooth=True, plot=plot, dpi=300, debug=debug,
                                                    save_plot=outpath_sub+'contrast_adi.pdf', verbose=verbose)
            if verbose:
                print("======= Completed Median-ADI =======")

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
                   tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size = self.fwhm, gauss_mode='conv')
               write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str  + '_conv.fits',
                          tmp, verbose=verbose)
           ### SNR map
           if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits')) and do_snr_map:
               tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits',verbose=verbose)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=self.nproc, verbose=verbose)
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
                                    nproc=self.nproc, verbose=verbose)
                   tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
               write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits',
                          tmp, verbose=verbose)

           if verbose:
               print("======= Completed PCA Full Frame =======")

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
           if debug:
               array_der = np.zeros([ntest_pcs, PCA_ADI_cube.shape[0], PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
               array_out = np.zeros([ntest_pcs, PCA_ADI_cube.shape[0], PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
           if do_snr_map_opt:
               tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])

           for pp, npc in enumerate(test_pcs_ann):
               if debug: # saves residuals and median
                    array_out[pp], array_der[pp], tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                         radius_int=mask_IWA_px, fwhm=self.fwhm, asize=ann_sz*self.fwhm,
                                         n_segments=1, delta_rot=delta_rot, delta_sep=(0.1, 1), ncomp=int(npc),
                                         svd_mode=svd_mode, nproc=self.nproc, min_frames_lib=max(npc, 10),
                                         max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                         interpolation='lanczos4', collapse='median', ifs_collapse_range='all',
                                         full_output=debug, verbose=verbose)
               else:
                   tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                             radius_int=mask_IWA_px, fwhm=self.fwhm, asize=ann_sz * self.fwhm,
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
           if debug:
               write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits', array_out, verbose=verbose)
               write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits', array_der, verbose=verbose)
           if do_snr_map_opt:
               write_fits(outpath_sub + 'neg_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp_tmp_tmp,verbose=verbose)

           ### Convolution
           if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits'):
               tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
               for nn in range(tmp.shape[0]):
                   tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size=self.fwhm, gauss_mode='conv')
               write_fits(outpath_sub+'final_PCA-ADI_ann_'+test_pcs_str+'_conv.fits',
                          tmp, verbose=verbose)

           ### SNR map
           if (not isfile(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits')) and do_snr_map:
               tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=self.nproc, verbose=verbose)
                   tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
               write_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap.fits', tmp, verbose=verbose)
           ### SNR map optimized
           if (not isfile(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits'))and do_snr_map_opt:
               tmp = open_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'.fits', verbose=verbose)
               tmp_tmp = open_fits(outpath_sub +'neg_PCA-ADI_ann_'+test_pcs_str+'.fits',verbose=verbose)
               for pp in range(ntest_pcs):
                   tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=plot, array2=tmp_tmp_tmp_tmp[pp], nproc=self.nproc,
                                    verbose=verbose)
                   tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
               write_fits(outpath_sub +'final_PCA-ADI_ann_'+test_pcs_str+'_snrmap_opt.fits',tmp, verbose=verbose)

           if verbose:
               print("======= Completed PCA Annular =======")

    def do_negfc(self,do_firstguess=True, guess_xy=None,mcmc_negfc=True, ncomp=1, algo=pca_annular,
                 nwalkers_ini=120, niteration_min = 25, niteration_limit=10000, weights=False, save_plot=True,verbose=True):
        """
        Module for estimating the location and flux of a planet.

        Using a first guess from the (x,y) coordinates in pixels for the planet, we can estimate a preliminary guess for
        the position and flux for each planet.

        If using MCMC, runs an affine invariant MCMC sampling algorithm in order to determine the position and the flux
        of the planet using the 'Negative Fake Companion' technique. The result of this procedure is a chain with the
        samples from the posterior distributions of each of the 3 parameters.

        Parameters:
        ***********
        do_firstguess : bool
            whether to determine a first guess for the position and the flux of a planet (not NEGFC)
        guess_xy : tuple
            if do_firstguess = True, first estimate of the source location to be provdied to firstguess()
        mcmc_negfc : bool
            whether to run MCMC NEGFC sampling (computationally intensive)
        ncomp : int, default 1
            number of prinicple components to subtract
        algo : 'pca_annulus', 'pca_annular', 'pca'. default 'pca_annular'
            select which routine to be used to model and subtract the stellar PSF
        nwalkers_ini : int, default 120
            for MCMC, the number of Goodman & Weare 'walkers'
        niteration_min : int, default 25
            for MCMC, the simulation will run at least this number of steps per walker
        niteration_limit : int, default 10000
            for MCMC, stops simulation if this many steps run without having reached the convergence criterion
        weights : bool
            for MCMC, should only be used on unsaturated datasets, where the flux of the star can be measured in each
            image of the cube
        save_plot : bool, default True
            the MCMC results are pickled and saved to the outpath
        verbose : bool
            prints more output when True
        """

        print("======= Starting NEGFC....=======")
        if guess_xy==None:
            raise ValueError("Enter an approximate location into guess_xy!")

        outpath_sub = self.outpath + "negfc/"

        if not isdir(self.outpath):
            os.system("mkdir " + self.outpath)
        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath))
            print('Output path is {}'.format(outpath_sub))

        source = self.dataset_dict['source']
        tn_shift = 0.58 # Milli et al. 2017, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'  # template name for input master cube
        derot_ang_name = 'derot_angles.fits'  # template name for corresponding input derotation angles
        psfn_name = "master_unsat_psf_norm.fits"  # normalised PSF

        ADI_cube = open_fits(self.inpath + ADI_cube_name.format(source), verbose=verbose)
        derot_angles = open_fits(self.inpath + derot_ang_name, verbose=verbose) + tn_shift
        psfn = open_fits(self.inpath + psfn_name, verbose=verbose)

        if algo == 'pca_annular':
            label_pca = 'pca_annular'
            algo = pca_annular
        elif algo == 'pca_annulus':
            label_pca = 'pca_annulus'
            algo = pca_annulus
        elif algo == 'pca':
            label_pca = 'pca'
            algo = pca
        else:
            raise ValueError("Invalid algorithm. Select either pca_annular, pca_annulus or pca!")
        opt_npc = ncomp
        ap_rad = 1 * self.fwhm
        f_range = np.geomspace(0.1, 201, 40)

        if not isfile(outpath_sub+label_pca+"_npc{}_simplex_results.fits".format(opt_npc)) and do_firstguess:
            ini_state = firstguess(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                   planets_xy_coord=guess_xy, fwhm=self.fwhm,
                                   annulus_width=12, aperture_radius=ap_rad, cube_ref=None,
                                   svd_mode='lapack', scaling=None, fmerit='stddev', imlib='opencv',
                                   interpolation='lanczos4', collapse='median', p_ini=None,
                                   transmission=None, algo=algo,
                                   f_range=f_range, simplex=True, simplex_options=None, plot=False,
                                   verbose=verbose, save=save_plot)
                                    # when p_ini is set to None, it gets the value of planets_xy_coord
            ini_state = np.array([ini_state[0][0], ini_state[1][0], ini_state[2][0]])

            write_fits(outpath_sub+label_pca+"_npc{}_simplex_results.fits".format(opt_npc), ini_state,verbose=verbose)

        if not isfile(outpath_sub + "MCMC_results") and mcmc_negfc:
            ini_state = open_fits(outpath_sub + label_pca + "_npc{}_simplex_results.fits".format(opt_npc), verbose=verbose)

            if weights:
                flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"  # flux in a FWHM aperture found in calibration
                weights = open_fits(self.inpath + flux_psf_name, verbose=verbose)[1]  # scaled fwhm flux is the second entry
            else:
                weights = None

            cy, cx = frame_center(ADI_cube[0])
            dy_pl = guess_xy[0][1] - cy
            dx_pl = guess_xy[0][0] - cx
            r_pl = np.sqrt(np.power(dx_pl, 2) + np.power(dy_pl, 2))
            theta_pl = (np.rad2deg(np.arctan2(dy_pl, dx_pl))) % 360
            print("Estimated (r, PA) = ({:.1f},{:.1f})".format(r_pl, theta_pl))
            delta_theta_min = np.rad2deg(np.arctan(4./r_pl)) #at least the angle corresponding to 2 azimuthal pixels
            delta_theta = max(delta_theta_min,5.)
            asize = 3 * self.fwhm
            bounds = [(max(r_pl - asize / 2., 1), r_pl + asize / 2.),  # radius
                      (theta_pl - delta_theta, theta_pl + delta_theta),  # angle
                      (0, 5 * abs(ini_state[2]))]

            mcmc_negfc_sampling(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                initial_state=ini_state, fwhm=self.fwhm, weights=weights,
                                annulus_width=12, aperture_radius=ap_rad, cube_ref=None,
                                svd_mode='lapack', scaling=None, fmerit='stddev',
                                imlib='opencv', interpolation='lanczos4', transmission=None,
                                collapse='median', nwalkers=nwalkers_ini, bounds=bounds, a=2.0,
                                ac_c=50, mu_sigma=(0, 1),
                                burnin=0.3, rhat_threshold=1.01, rhat_count_threshold=1, conv_test='ac',
                                # use autocorrelation to ensure sufficient sampling. sample around the aea of best likelihood to make distribution
                                niteration_min=niteration_min, niteration_limit=niteration_limit,
                                niteration_supp=0, check_maxgap=50, nproc=self.nproc, algo=algo,
                                output_dir=outpath_sub,
                                output_file="MCMC_results", display=False, verbosity=2,
                                save=save_plot)
