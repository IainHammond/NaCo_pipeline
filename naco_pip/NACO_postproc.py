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
from vip_hci.metrics import snrmap, contrast_curve, normalize_psf, cube_inject_companions
from vip_hci.negfc import mcmc_negfc_sampling, firstguess, show_walk_plot, show_corner_plot, confidence
from vip_hci.pca import pca, pca_annular, pca_annulus
from vip_hci.preproc import cube_crop_frames
from vip_hci.medsub import median_sub
from vip_hci.var import mask_circle, frame_filter_lowpass, frame_center


class preproc_dataset:  # this class is for post-processing of the pre-processed data
    def __init__(self, inpath, outpath, dataset_dict, nproc, npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        try:
            self.fwhm = open_fits(self.inpath + 'fwhm.fits', verbose=False)[0]  # fwhm is first entry
        except:
            print("Alert: No FWHM file found. Setting to median value of 4.2")
            self.fwhm = 4.2
        self.dataset_dict = dataset_dict
        self.pixel_scale = dataset_dict['pixel_scale']
        if not isdir(self.outpath):
            os.system("mkdir " + self.outpath)

    def postprocessing(self, do_adi=True, do_adi_contrast=True, do_pca_full=True, do_pca_ann=True, cropped=True,
                       do_snr_map=True, do_snr_map_opt=True, delta_rot=(0.5, 3), mask_IWA=1, overwrite=True, plot=True,
                       verbose=True, debug=False):
        """ 
        For post processing the master cube via median ADI, full frame PCA-ADI, or annular PCA-ADI. Includes constrast
        curves and SNR maps.

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
        overwrite : bool, default True
            whether to overwrite pre-exisiting output files from previous reductions
        plot : bool
            Whether to save plots to the output path (PDF file, print quality)
        verbose : bool
            prints more output when True                 
        debug : bool, default is False
            Saves extra output files
        """

        # make directories if they don't exist
        print("======= Starting post-processing....=======")
        outpath_sub = self.outpath + "sub_npc{}/".format(self.npc)

        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath))
            print('Output path is {}'.format(outpath_sub))

        source = self.dataset_dict['source']
        tn_shift = 0.58  # Milli et al. 2017, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'  # template name for input master cube
        derot_ang_name = 'derot_angles.fits'  # template name for corresponding input derotation angles
        ADI_cube = open_fits(self.inpath + ADI_cube_name.format(source), verbose=verbose)
        derot_angles = open_fits(self.inpath + derot_ang_name, verbose=verbose) + tn_shift

        if do_adi_contrast:
            psfn_name = "master_unsat_psf_norm.fits"  # normalised PSF
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"  # flux in a FWHM aperture found in calibration
            psfn = open_fits(self.inpath + psfn_name, verbose=verbose)
            starphot = open_fits(self.inpath + flux_psf_name, verbose=verbose)[1]  # scaled fwhm flux is the second entry

        mask_IWA_px = mask_IWA * self.fwhm
        if verbose:
            print("adopted mask size: {:.0f}".format(mask_IWA_px))

        ann_sz = 3  # if PCA-ADI in a single annulus or in concentric annuli, this is the size of the annulus/i in FWHM
        svd_mode = 'lapack'  # python package used for Singular Value Decomposition for PCA reductions
        n_randsvd = 3  # if svd package is set to 'randsvd' number of times we do PCA rand-svd, before taking the
        # median of all results (there is a risk of significant self-subtraction when just doing it once)
        ref_cube = None  # if any, load here a centered calibrated cube of reference star observations - would then be
        # used for PCA instead of the SCI cube itself

        # TEST number of principal components
        # PCA-FULL
        if do_pca_full:
            test_pcs_full = list(range(1, self.npc + 1))
        # PCA-ANN
        if do_pca_ann:
            test_pcs_ann = list(range(1, self.npc + 1))  # needs a cropped cube

        ######################### Simple ADI ###########################
        if do_adi:
            if not isfile(outpath_sub + 'final_ADI_simple.fits') or overwrite:
                if debug:  # saves the residuals
                    tmp, _, tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                                 radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                                 full_output=debug, verbose=verbose)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'TMP_ADI_simple_cube_der.fits', tmp, verbose=verbose)

                # make median combination of the de-rotated cube.
                tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm,
                                     radius_int=0, asize=ann_sz, delta_rot=delta_rot,
                                     full_output=False, verbose=verbose)
                tmp_tmp = mask_circle(tmp_tmp, mask_IWA_px)  # we mask the IWA
                write_fits(outpath_sub + 'final_ADI_simple.fits', tmp_tmp, verbose=verbose)

            ## SNR map
            if (not isfile(outpath_sub + 'final_ADI_simple_snrmap.fits') or overwrite) and do_snr_map:
                tmp = open_fits(outpath_sub + 'final_ADI_simple.fits', verbose=verbose)
                tmp = mask_circle(tmp, mask_IWA_px)
                tmp_tmp = snrmap(tmp, self.fwhm, nproc=self.nproc, verbose=debug)
                write_fits(outpath_sub + 'final_ADI_simple_snrmap.fits', tmp_tmp, verbose=verbose)

            ## Contrast curve
            if (not isfile(outpath_sub + 'contrast_adi.pdf') or overwrite) and do_adi_contrast:
                _ = contrast_curve(ADI_cube, derot_angles, psfn, self.fwhm, pxscale=self.pixel_scale,
                                   starphot=starphot, algo=median_sub, sigma=5., nbranch=1, theta=0,
                                   inner_rad=1, wedge=(0, 360), student=True, transmission=None,
                                   smooth=True, plot=plot, dpi=300, debug=debug,
                                   save_plot=outpath_sub + 'contrast_adi.pdf', verbose=verbose)
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
                                              svd_mode='randsvd', scaling=None, mask_center_px=mask_IWA_px,
                                              delta_rot=delta_rot, fwhm=self.fwhm, collapse='median', check_memory=True,
                                              full_output=False, verbose=verbose)
                    tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                else:
                    if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite:
                        tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube,
                                          scale_list=None, ncomp=int(npc),
                                          svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                          delta_rot=delta_rot, fwhm=self.fwhm, collapse='median', check_memory=True,
                                          full_output=False, verbose=verbose)
                    if (not isfile(outpath_sub + 'final_PCA-ADI_full_' +test_pcs_str+'_snrmap_opt.fits') or overwrite) \
                            and do_snr_map_opt:
                        tmp_tmp_tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=-derot_angles, cube_ref=ref_cube,
                                                  scale_list=None, ncomp=int(npc),
                                                  svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                                  delta_rot=delta_rot, fwhm=self.fwhm, collapse='median',
                                                  check_memory=True,
                                                  full_output=False, verbose=verbose)
            if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite:
                write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', tmp_tmp, verbose=verbose)

            if (not isfile(outpath_sub + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite) and do_snr_map_opt:
                write_fits(outpath_sub + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',
                           tmp_tmp_tmp_tmp, verbose=verbose)
            ### Convolution
            if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits') or overwrite:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=verbose)
                for nn in range(tmp.shape[0]):
                    tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size=self.fwhm, gauss_mode='conv')
                write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits', tmp, verbose=verbose)

            ### SNR map
            if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits') or overwrite) \
                    and do_snr_map:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=verbose)
                for pp in range(ntest_pcs):
                    tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=self.nproc, verbose=debug)
                    tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits', tmp, verbose=verbose)

            ### SNR map optimized
            if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) \
                    and do_snr_map_opt:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=verbose)
                for pp in range(ntest_pcs):
                    tmp[pp] = snrmap(tmp[pp], self.fwhm, array2=tmp_tmp_tmp_tmp[pp], incl_neg_lobes=False,
                                     nproc=self.nproc, verbose=debug)
                    tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits', tmp,
                           verbose=verbose)

            if verbose:
                print("======= Completed PCA Full Frame =======")

        ######################## PCA-ADI annular #######################
        if do_pca_ann:
            if cropped == False:
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
                if debug and ((not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits') and
                               not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits'))
                              or overwrite):
                    # saves residuals and median if debug is true and they either dont exist or are to be overwritten
                    array_out[pp], array_der[pp], tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles,
                                                                            cube_ref=ref_cube, scale_list=None,
                                                                            radius_int=mask_IWA_px, fwhm=self.fwhm,
                                                                            asize=ann_sz * self.fwhm,
                                                                            n_segments=1, delta_rot=delta_rot, ncomp=int(npc),
                                                                            svd_mode=svd_mode, nproc=self.nproc,
                                                                            min_frames_lib=max(npc, 10),
                                                                            max_frames_lib=200, tol=1e-1, scaling=None,
                                                                            imlib='opencv',
                                                                            interpolation='lanczos4', collapse='median',
                                                                            ifs_collapse_range='all',
                                                                            full_output=debug, verbose=verbose)
                else:
                    if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits') or overwrite:
                        tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                                  radius_int=mask_IWA_px, fwhm=self.fwhm, asize=ann_sz * self.fwhm,
                                                  n_segments=1, delta_rot=delta_rot, ncomp=int(npc),
                                                  svd_mode=svd_mode, nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                  max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                  interpolation='lanczos4', collapse='median', ifs_collapse_range='all',
                                                  full_output=False, verbose=verbose)
                if (not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                    tmp_tmp_tmp_tmp[pp] = pca_annular(PCA_ADI_cube, -derot_angles, cube_ref=ref_cube,
                                                      scale_list=None, radius_int=mask_IWA_px, fwhm=self.fwhm,
                                                      asize=ann_sz * self.fwhm, n_segments=1, delta_rot=delta_rot,
                                                      ncomp=int(npc), svd_mode=svd_mode,
                                                      nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                      max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                      interpolation='lanczos4', collapse='median',
                                                      ifs_collapse_range='all', full_output=False, verbose=verbose)
            if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits') or overwrite:
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp, verbose=verbose)
            if debug and ((not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits') and
                           not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits')) or overwrite):
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits', array_out, verbose=verbose)
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits', array_der, verbose=verbose)

            if (not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                write_fits(outpath_sub + 'neg_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp_tmp_tmp, verbose=verbose)

            ### Convolution
            if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits') or overwrite:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=verbose)
                for nn in range(tmp.shape[0]):
                    tmp[nn] = frame_filter_lowpass(tmp[nn], fwhm_size=self.fwhm, gauss_mode='conv')
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits', tmp, verbose=verbose)

            ### SNR map
            if (not isfile(
                    outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap.fits') or overwrite) and do_snr_map:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=verbose)
                for pp in range(ntest_pcs):
                    tmp[pp] = snrmap(tmp[pp], self.fwhm, nproc=self.nproc, verbose=debug)
                    tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap.fits', tmp, verbose=verbose)
            ### SNR map optimized
            if (not isfile(
                    outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=verbose)
                for pp in range(ntest_pcs):
                    tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=plot, array2=tmp_tmp_tmp_tmp[pp], nproc=self.nproc,
                                     verbose=debug)
                    tmp[pp] = mask_circle(tmp[pp], mask_IWA_px)
                write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits', tmp, verbose=verbose)

            if verbose:
                print("======= Completed PCA Annular =======")

    def do_negfc(self, do_firstguess=True, guess_xy=None, mcmc_negfc=True, inject_neg=True, ncomp=1, algo='pca_annular',
                 nwalkers_ini=120, niteration_min=25, niteration_limit=10000, delta_rot=(0.5, 3), weights=False,
                 coronagraph=False, overwrite=True, save_plot=True, verbose=True):
        """
        Module for estimating the location and flux of a planet. A sub-folder 'negfc' is created for storing all
        output files.

        Using a first guess from the (x,y) coordinates in pixels for the planet, we can estimate a preliminary guess for
        the position and flux for each planet using a Nelder-Mead minimization. Saves the r, theta and flux

        If using MCMC, runs an affine invariant MCMC sampling algorithm in order to determine the position and the flux
        of the planet using the 'Negative Fake Companion' technique. The result of this procedure is a chain with the
        samples from the posterior distributions of each of the 3 parameters.

        Finally, we can inject a negative flux of the planet found in MCMC into the original dataset and apply
        post processing to determine the residuals in the data.

        Parameters:
        ***********
        do_firstguess : bool
            whether to determine a first guess for the position and the flux of a planet (not NEGFC)
        guess_xy : tuple
            if do_firstguess=True, this estimate of the source (x,y) location to be provided to firstguess(). Note
            Python's zero-based indexing
        mcmc_negfc : bool
            whether to run MCMC NEGFC sampling (computationally intensive)
        inject_neg : bool
            whether to inject negative flux of the planet and post process the data without signal from the planet
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
        delta_rot : tuple
            same as for postprocessing module. Threshold rotation angle for PCA annular
        weights : bool
            should only be used on unsaturated datasets, where the flux of the star can be measured in each image of
            the cube. Applies a correction to each frame to account for variability of the adaptive optics, and hence
            flux from the star (reference Christiaens et al. 2021 2021MNRAS.502.6117C)
        coronagraph : bool
            for MCMC and injecting a negative companion, True if the observation utilised a coronagraph (AGPM).
            The known radial transmission of the NACO+AGPM coronagraph will be used
        overwrite : bool, default True
            whether to run a module and overwrite results if they already exist
        save_plot : bool, default True
            for firstguess the chi2 vs. flux plot is saved. For MCMC results are pickled and saved to the outpath
            along with corner plots
        verbose : bool
            prints more output and interediate files when True
        """

        print("======= Starting NEGFC....=======")
        if guess_xy is None and do_firstguess is True:
            raise ValueError("Enter an approximate location into guess_xy!")

        if weights is True and coronagraph is True:
            raise ValueError("Dataset cannot be both non-coronagraphic and coronagraphic!!")

        outpath_sub = self.outpath + "negfc/"

        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath))
            print('Output path is {}'.format(outpath_sub))

        source = self.dataset_dict['source']
        tn_shift = 0.58  # Milli et al. 2017, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'  # template name for input master cube
        derot_ang_name = 'derot_angles.fits'  # template name for corresponding input derotation angles
        psfn_name = "master_unsat_psf_norm.fits"  # normalised PSF

        ADI_cube = open_fits(self.inpath + ADI_cube_name.format(source), verbose=verbose)
        derot_angles = open_fits(self.inpath + derot_ang_name, verbose=verbose) + tn_shift
        psfn = open_fits(self.inpath + psfn_name, verbose=verbose)
        ref_cube = None

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
        asize = 3 * self.fwhm

        if weights:
            nfr = ADI_cube.shape[0]  # number of frames
            star_flux = np.zeros([nfr])  # for storing the star flux found in each frame
            crop_sz_tmp = min(int(10 * self.fwhm),
                              ADI_cube.shape[1] - 2)  # crop around star, either 10*FWHM or size - 2
            if crop_sz_tmp % 2 == 0:  # if it's not even, crop
                crop_sz_tmp -= 1
            for ii in range(nfr):
                _, star_flux[ii], _ = normalize_psf(ADI_cube[ii], fwhm=self.fwhm, size=crop_sz_tmp,
                                                    full_output=True)  # get star flux in 1*FWHM
            weights = star_flux / np.median(star_flux)
            star_flux = np.median(star_flux)  # for use after MCMC when turning the chain into contrast
        else:
            weights = None
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"  # flux in a FWHM aperture found in calibration
            star_flux = open_fits(self.inpath + flux_psf_name, verbose=verbose)[1]  # scaled fwhm flux

        if coronagraph:  # radial transmission of the coronagraph, 2 columns (pixels from centre, off-axis transmission)
        #  data provided by Valentin Christiaens. First entry in both columns was not zero, but VIP adds it in anyway
            transmission = np.array([[0, 3.5894626e-10, 5.0611424e-01, 1.0122285e+00, 1.5183427e+00,
                                    2.0244570e+00, 2.5305712e+00, 3.0366855e+00, 3.5427995e+00,
                                    4.0489140e+00, 4.5550284e+00, 5.0611424e+00, 5.5672565e+00,
                                    6.0733705e+00, 6.5794849e+00, 7.0855989e+00, 7.5917134e+00,
                                    8.6039419e+00, 9.1100569e+00, 9.6161709e+00, 1.0628398e+01,
                                    1.1134513e+01, 1.2146742e+01, 1.2652856e+01, 1.3665085e+01,
                                    1.4677314e+01, 1.6195656e+01, 1.7207884e+01, 1.8220114e+01,
                                    1.9738455e+01, 2.1256796e+01, 2.2775141e+01, 2.4293484e+01,
                                    2.6317940e+01, 2.8342396e+01, 3.0366854e+01, 3.2897423e+01,
                                    3.4921883e+01, 3.7452454e+01, 4.0489140e+01, 4.3525822e+01,
                                    4.6562508e+01, 5.0105309e+01, 5.4154221e+01, 5.7697018e+01,
                                    6.2252052e+01, 6.6807076e+01, 7.1868225e+01],
                                    [0, 6.7836474e-05, 3.3822558e-03, 1.7766271e-02, 5.2646037e-02,
                                    1.1413762e-01, 1.9890217e-01, 2.9460809e-01, 3.8605216e-01,
                                    4.6217495e-01, 5.1963091e-01, 5.6185508e-01, 5.9548348e-01,
                                    6.2670821e-01, 6.5912777e-01, 6.9335037e-01, 7.2783405e-01,
                                    7.8866738e-01, 8.1227022e-01, 8.3128709e-01, 8.5912752e-01,
                                    8.6968899e-01, 8.8677746e-01, 8.9409947e-01, 9.0848678e-01,
                                    9.2426234e-01, 9.4704604e-01, 9.5787460e-01, 9.6538281e-01,
                                    9.7379774e-01, 9.8088801e-01, 9.8751044e-01, 9.9255627e-01,
                                    9.9640906e-01, 9.9917024e-01, 1.0009050e+00, 1.0021056e+00,
                                    1.0026742e+00, 1.0027454e+00, 1.0027291e+00, 1.0023015e+00,
                                    1.0016677e+00, 1.0009446e+00, 1.0000550e+00, 9.9953103e-01,
                                    9.9917012e-01, 9.9915260e-01, 9.9922234e-01]])
        else:
            transmission = None

        if (not isfile(outpath_sub + label_pca + "_npc{}_simplex_results.fits".format(opt_npc)) or overwrite) and do_firstguess:

            # find r, theta based on the provided estimate location
            cy, cx = frame_center(ADI_cube[0])
            dy_pl = guess_xy[0][1] - cy
            dx_pl = guess_xy[0][0] - cx
            r_pl = np.sqrt(np.power(dx_pl, 2) + np.power(dy_pl, 2))  # pixel distance to the guess location
            theta_pl = (np.rad2deg(np.arctan2(dy_pl, dx_pl))) % 360  # theta (angle) to the guess location
            print("Estimated (r, theta) before first guess = ({:.1f},{:.1f})".format(r_pl, theta_pl))

            ini_state = firstguess(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                   planets_xy_coord=guess_xy, fwhm=self.fwhm,
                                   annulus_width=12, aperture_radius=ap_rad, cube_ref=ref_cube,
                                   svd_mode='lapack', scaling=None, fmerit='stddev', imlib='opencv',
                                   interpolation='lanczos4', collapse='median', p_ini=None,
                                   transmission=transmission, weights=weights, algo=algo,
                                   f_range=f_range, simplex=True, simplex_options=None, plot=save_plot,
                                   verbose=verbose, save=save_plot)
            # when p_ini is set to None, it gets the value of planets_xy_coord
            ini_state = np.array([ini_state[0][0], ini_state[1][0], ini_state[2][0]])

            write_fits(outpath_sub + label_pca + "_npc{}_simplex_results.fits".format(opt_npc), ini_state,
                       verbose=verbose)  # saves r, theta and flux. No print statement as firstguess() does that for us

        if (not isfile(outpath_sub + "MCMC_results") or overwrite) and mcmc_negfc:
            ini_state = open_fits(outpath_sub + label_pca + "_npc{}_simplex_results.fits".format(opt_npc),
                                  verbose=verbose)

            delta_theta_min = np.rad2deg(np.arctan(4./ r_pl))  # at least the angle corresponding to 2 azimuthal pixels
            delta_theta = max(delta_theta_min, 5.)
            bounds = [(max(r_pl - asize / 2., 1), r_pl + asize / 2.),  # radius
                      (theta_pl - delta_theta, theta_pl + delta_theta),  # angle
                      (0, 5 * abs(ini_state[2]))]

            if ini_state[0] < bounds[0][0] or ini_state[0] > bounds[0][1] or ini_state[1] < bounds[1][0] or \
                    ini_state[1] > bounds[1][1] or ini_state[2] < bounds[2][0] or ini_state[2] > bounds[2][1]:
                print("!!! WARNING: simplex results not in original bounds - NEGFC simplex MIGHT HAVE FAILED !!!")
                ini_state = np.array([r_pl, theta_pl, abs(ini_state[2])])

            if verbose is True:
                verbosity = 2
                print('MCMC NEGFC sampling is about to begin...')
            else:
                verbosity = 0

            final_chain = mcmc_negfc_sampling(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                              initial_state=ini_state, fwhm=self.fwhm, weights=weights,
                                              annulus_width=12, aperture_radius=ap_rad, cube_ref=ref_cube,
                                              svd_mode='lapack', scaling=None, fmerit='stddev',
                                              imlib='opencv', interpolation='lanczos4', transmission=transmission,
                                              collapse='median', nwalkers=nwalkers_ini, bounds=bounds, a=2.0,
                                              ac_c=50, mu_sigma=(0, 1),
                                              burnin=0.3, rhat_threshold=1.01, rhat_count_threshold=1, conv_test='ac',
                                              # use autocorrelation 'ac' to ensure sufficient sampling. sample around
                                              # the area of best likelihood to make distribution
                                              niteration_min=niteration_min, niteration_limit=niteration_limit,
                                              niteration_supp=0, check_maxgap=50, nproc=self.nproc, algo=algo,
                                              output_dir=outpath_sub,
                                              output_file="MCMC_results", display=False, verbosity=verbosity,
                                              save=save_plot)

            final_chain[:, :, 2] = final_chain[:, :, 2] / star_flux  # dividing by the star flux converts to a contrast
            show_walk_plot(final_chain, save=save_plot, output_dir=outpath_sub)
            show_corner_plot(final_chain, burnin=0.5, save=save_plot, output_dir=outpath_sub)

            # determine the highly probable value for each model parameter and the 1-sigma confidence interval
            isamples_flat = final_chain[:,int(final_chain.shape[1]//(1/0.3)):,:].reshape((-1,3))  # 0.3 is the burnin
            vals, err = confidence(isamples_flat, cfd=68.27, bins=100, gaussian_fit=False, weights=weights,
                                   verbose=verbose, save=True, output_dir=outpath_sub, filename='confidence.txt',
                                   plsc=self.pixel_scale)

            labels = ['r', 'theta', 'f']
            mcmc_res = np.zeros([3,3])
            # pull the values and confidence interval out for saving
            for i in range(3):
                mcmc_res[i,0] = vals[labels[i]]
                mcmc_res[i,1] = err[labels[i]][0]
                mcmc_res[i,2] = err[labels[i]][1]
            write_fits(outpath_sub + 'mcmc_results.fits', mcmc_res)

            # now gaussian fit
            gvals, gerr = confidence(isamples_flat, cfd=68.27, bins=100, gaussian_fit=True, weights=weights,
                                     verbose=verbose, save=True, output_dir=outpath_sub,filename='confidence_gauss.txt',
                                     plsc=self.pixel_scale)

            mcmc_res = np.zeros([3,2])
            for i in range(3):
                mcmc_res[i,0] = gvals[i]
                mcmc_res[i,1] = gerr[i]
            write_fits(outpath_sub + 'mcmc_results_gauss.fits', mcmc_res)

        if inject_neg:
            pca_res = np.zeros([ADI_cube.shape[1], ADI_cube.shape[2]])
            pca_res_emp = pca_res.copy()
            planet_params = open_fits(outpath_sub+'mcmc_results.fits')
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"
            star_flux = open_fits(self.inpath + flux_psf_name, verbose=verbose)[1]

            ADI_cube_emp = cube_inject_companions(ADI_cube, psfn, derot_angles,
                                                  flevel=-planet_params[2, 0] * star_flux, plsc=self.pixel_scale,
                                                  rad_dists=[planet_params[0, 0]],
                                                  n_branches=1, theta=planet_params[1, 0],
                                                  imlib='opencv', interpolation='lanczos4',
                                                  verbose=verbose, transmission=transmission)
            write_fits(outpath_sub+'ADI_cube_empty.fits', ADI_cube_emp)  # the cube with the negative flux injected

            if algo == pca_annular:
                radius_int = int(np.floor(r_pl-asize/2))  # asize is 3 * FWHM, rounds down. To skip the inner region
                # crop the cube to just larger than the annulus to improve the speed of PCA
                crop_sz = int(2*np.ceil(r_pl+asize+1))  # rounds up
                if not crop_sz % 2:  # make sure the crop is odd
                    crop_sz += 1
                if crop_sz < ADI_cube.shape[1] and crop_sz < ADI_cube.shape[2]:  # crop if crop_sz is smaller than cube
                    pad = int((ADI_cube.shape[1]-crop_sz)/2)
                    crop_cube = cube_crop_frames(ADI_cube, crop_sz, verbose=verbose)
                else:
                    crop_cube = ADI_cube  # dont crop if the cube is already smaller

                pca_res_tmp = pca_annular(crop_cube, derot_angles, cube_ref=ref_cube, radius_int=radius_int,
                                          fwhm=self.fwhm, asize=asize, delta_rot=delta_rot, ncomp=opt_npc,
                                          svd_mode='lapack', scaling=None, imlib='opencv', interpolation='lanczos4',
                                          nproc=self.nproc, min_frames_lib=max(opt_npc, 10), verbose=verbose,
                                          full_output=False)

                pca_res = np.pad(pca_res_tmp, pad, mode='constant', constant_values=0)
                write_fits(outpath_sub + 'pca_annular_res_npc{}.fits'.format(opt_npc), pca_res)

                # emp
                if crop_sz < ADI_cube_emp.shape[1] and crop_sz < ADI_cube_emp.shape[2]:
                    pad = int((ADI_cube_emp.shape[1]-crop_sz)/2)
                    crop_cube = cube_crop_frames(ADI_cube_emp, crop_sz, verbose=verbose)
                else:
                    crop_cube = ADI_cube_emp
                del ADI_cube_emp
                del ADI_cube

                pca_res_tmp = pca_annular(crop_cube, derot_angles, cube_ref=ref_cube, radius_int=radius_int,
                                          fwhm=self.fwhm, asize=asize, delta_rot=delta_rot, ncomp=opt_npc,
                                          svd_mode='lapack', scaling=None, imlib='opencv', interpolation='lanczos4',
                                          nproc=self.nproc, min_frames_lib=max(opt_npc, 10), verbose=verbose,
                                          full_output=False)

                # pad again now
                pca_res_emp = np.pad(pca_res_tmp, pad, mode='constant', constant_values=0)
                write_fits(outpath_sub+'pca_annular_res_empty_npc{}.fits'.format(opt_npc), pca_res_emp)
