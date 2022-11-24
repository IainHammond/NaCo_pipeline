#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applies post-processing algorithms to the pre-processed cube

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['preproc_dataset']

import os
from os.path import isfile, isdir

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as Df
from pandas import read_csv

try:
    from vip_hci.config import time_ini, timing
    from vip_hci.psfsub import pca, pca_annular, pca_annulus, median_sub
    from vip_hci.fm import normalize_psf, cube_inject_companions, mcmc_negfc_sampling, firstguess, show_walk_plot, \
        show_corner_plot, confidence
except:
    from vip_hci.conf import time_ini, timing
    from vip_hci.medsub import median_sub
    from vip_hci.pca import pca, pca_annular, pca_annulus
    from vip_hci.metrics import normalize_psf, cube_inject_companions
    from vip_hci.negfc import mcmc_negfc_sampling, firstguess, show_walk_plot, show_corner_plot, confidence
    print('Attention: A newer version of VIP is available.', flush=True)
from vip_hci.fits import open_fits, write_fits
from vip_hci.metrics import snrmap, contrast_curve, snr
from vip_hci.preproc import cube_crop_frames
from vip_hci.var import mask_circle, frame_filter_lowpass, frame_center

mpl.use('Agg')


def find_nearest(array, value, output='index', constraint=None, n=1):
    """
    Function to find the indices, and optionally the values, of an array's n closest elements to a certain value.
    Possible outputs: 'index','value','both'
    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest element with a value greater than 'value', "floor" the opposite)
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")

    if constraint is None:
        fm = np.absolute(array - value)

    elif constraint == 'ceil':
        fm = array - value
        fm = fm[np.where(fm > 0)]
    elif constraint == 'floor':
        fm = -(array - value)
        fm = fm[np.where(fm > 0)]
    else:
        raise ValueError("Constraint not recognised")

    idx = fm.argsort()[:n]
    if n == 1:
        idx = idx[0]

    if output == 'index':
        return idx
    elif output == 'value':
        return array[idx]
    else:
        return array[idx], idx


class preproc_dataset:  # this class is for post-processing of the pre-processed data
    def __init__(self, inpath, outpath, dataset_dict, nproc, npc):
        self.inpath = inpath
        self.outpath = outpath
        self.nproc = nproc
        self.npc = npc
        try:
            self.fwhm = open_fits(self.inpath + 'fwhm.fits', verbose=False)[0]  # fwhm is first entry
        except:
            print("Alert: No FWHM file found. Setting to median value of 4.2", flush=True)
            self.fwhm = 4.2
        self.dataset_dict = dataset_dict
        self.pixel_scale = dataset_dict['pixel_scale']
        self.source = dataset_dict['source']
        self.details = dataset_dict['details']
        if not isdir(self.outpath):
            os.makedirs(self.outpath)

    def postprocessing(self, do_adi=True, do_adi_contrast=True, do_pca_full=True, do_pca_ann=True, fake_planet=False,
                       first_guess_skip=False, fcp_pos=[0.3], firstguess_pcs=[1, 21, 1], cropped=True, do_snr_map=True,
                       do_snr_map_opt=True, planet_pos=None, delta_rot=(0.5, 3), mask_IWA=1, coronagraph=True,
                       overwrite=True, verbose=True, debug=False):
        """ 
        For post processing the master cube via median ADI, full frame PCA-ADI, or annular PCA-ADI. Includes contrast
        curves, SNR maps and fake planet injection for optimizing the number of principal components.

        Natural progression:
        Run median-ADI and PCA-ADI in full frame and annular
            a) If a blob is found or if there is a known planet/companion, set source_xy to the approximate x-y
            coordinates and re-run PCA. Run NEGFC to determine it r, theta and flux
            b) If no blob is found, set fake_planet to True and create contrast curves

        Parameters:
        ----------
        do_adi : bool
            Whether to do a median-ADI processing
        do_adi_contrast : bool
            Whether to compute a simple contrast curve associated to median-ADI, saved as PDF (print quality)
        do_pca_full : bool
            Whether to apply PCA-ADI on full frame
        do_pca_ann : bool, default is False
            Whether to apply annular PCA-ADI (more computer intensive). Only runs if cropped=True
        fake_planet : bool
            Will inject fake planets into the cube to optimize number of principal components and produce contrast
            curves for PCA-ADI and/or PCA-ADI annular, depending on above. Increases run time - use a binned cube for a
            faster reduction, or non-binned for better contrast
        first_guess_skip : bool
            Will not complete a first contrast curve to determine the sensitivity required when injecting fake
            companions, but will still make a final contrast curve. For the purpose of decreasing run time, mostly
            relevant for non-binned data. Best contrast at each separation is a single loop over principal components.
        firstguess_pcs : list of 3 elements
            If fake planet is True and first guess is not skipped, this is the guess principal components to
            explore [start, stop, step] (zero based indexing)
        fcp_pos : list or 1D array, default [0.3]
            If fake planet is True and first guess is not skipped, this is the arcsecond separation for determine the
            optimal principal components
        cropped : bool
            Whether the master cube was cropped in pre-processing
        do_snr_map : bool
            Whether to compute an SNR map (warning: computer intensive); useful only when point-like features are seen
            in the image
        do_snr_map_opt : bool
            Whether to compute a non-conventional (more realistic) SNR map
        planet_pos : tuple, default None
            If there is a known (or suspected) companion/giant planet in the data, set this as its x-y coordinates in
            the post-processed data
        delta_rot : tuple
            Threshold in rotation angle used in pca_annular to include frames in the PCA library (in terms of FWHM).
            See description of pca_annular() for more details. Applied to full frame PCA-ADI in the case of a planet.
            Reduces the number of frames used to build the PCA library and increases run time but reduces companion
            self-subtraction, especially at close separations
        mask_IWA : int, default 1
            Size of the numerical mask that hides the inner part of post-processed images. Provided in terms of FWHM.
        coronagraph : bool, default is True
            Set to True if the observation used an AGPM coronagraph
        overwrite : bool, default True
            Whether to overwrite pre-existing output files from previous reductions
        verbose : bool
            Prints more output when True
        debug : bool, default is False
            Saves extra output files
        """
        print("======= Starting post-processing....=======", flush=True)

        # determine npcs, make directories if they don't exist
        if isinstance(self.npc, int):  # if just an integer or tuple of length 1 is provided
            test_pcs = [self.npc]  # just a single number of PCs
            outpath_sub = self.outpath + "sub_npc{}/".format(test_pcs[0])
        if isinstance(self.npc, list) or isinstance(self.npc, tuple):
            if len(self.npc) == 2:  # start and stop PCs
                test_pcs = list(range(self.npc[0], self.npc[1] + 1, 1))
            elif len(self.npc) == 3:  # start, stop and step PCs
                test_pcs = list(range(self.npc[0], self.npc[1] + 1, self.npc[2]))
            outpath_sub = self.outpath + "sub_npc{}-{}/".format(test_pcs[0], test_pcs[-1])
        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath), flush=True)
            print('Output path is {}'.format(outpath_sub), flush=True)

        if isinstance(delta_rot, list):
            delta_rot = tuple(delta_rot)
        if isinstance(planet_pos, list):
            planet_pos = tuple(planet_pos)

        source = self.dataset_dict['source']
        tn_shift = 0.572  # ± 0.178 Launhardt et al. 2020, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'  # template name for input master cube
        derot_ang_name = 'derot_angles.fits'  # template name for corresponding input derotation angles
        ADI_cube = open_fits(self.inpath + ADI_cube_name.format(source), verbose=verbose)
        derot_angles = open_fits(self.inpath + derot_ang_name, verbose=verbose) + tn_shift
        cy, cx = frame_center(ADI_cube[0], verbose=debug)
        svd_mode = 'lapack'  # can be changed to a different method for SVD. Note randsvd is not really supported

        if do_adi_contrast or fake_planet:
            psfn_name = "master_unsat_psf_norm.fits"  # normalised PSF
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"  # flux in a FWHM aperture found in calibration
            psfn = open_fits(self.inpath + psfn_name, verbose=verbose)
            starphot = open_fits(self.inpath+flux_psf_name, verbose=verbose)[1]  # scaled fwhm flux is the second entry
            nbranch = 6
            nspi = 6  # number of fake companion spirals, or just number of PAs to test if only one radii
            fc_snr = 10  # SNR of injected companions
            injection_fac = 1  # * 3 sigma contrast
            th_step = 360 / nspi  # PA separation between companions determined by number of companions
            if isinstance(fcp_pos, list):
                fcp_pos = np.array(fcp_pos)
            rad_arr = fcp_pos / self.pixel_scale  # separation in px
            while rad_arr[-1] >= ADI_cube.shape[-1]:  # ensure largest injection radius is not bigger than the frame
                rad_arr = rad_arr[:-1]
            nfcp = len(rad_arr)  # number of radii/separations to inject companions
            # radial transmission of the AGPM, 2 columns (pixels from centre, off-axis transmission)
            if coronagraph:
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
        mask_IWA_px = mask_IWA * self.fwhm
        if verbose:
            print("adopted mask size: {:.0f}".format(mask_IWA_px), flush=True)

        ann_sz = 3  # if PCA-ADI in concentric annuli, this is the size of the annulus/i in FWHM
        ref_cube = None  # if any, load here a centered calibrated cube of reference star observations - would then be
        # used for PCA instead of the SCI cube itself

        # TEST number of principal components
        # PCA-FULL
        if do_pca_full:
            test_pcs_full = test_pcs
        # PCA-ANN
        if do_pca_ann:
            if not cropped:  # needs a cropped cube
                raise ValueError('PCA-ADI annular requires a cropped cube from pre-processing!')
            test_pcs_ann = test_pcs
        # Contrast estimation
        if fake_planet and not first_guess_skip:
            firstguess_pcs = list(range(firstguess_pcs[0], firstguess_pcs[1], firstguess_pcs[2]))

        ######################### Simple ADI ###########################
        if do_adi:
            if verbose:
                print("======= Starting Median-ADI =======", flush=True)
            if not isfile(outpath_sub + 'final_ADI_simple.fits') or overwrite:
                if debug:  # saves the residuals
                    tmp, _, _ = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm, delta_rot=delta_rot,
                                           full_output=debug, verbose=verbose, nproc=self.nproc)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'TMP_ADI_simple_cube_der.fits', tmp, verbose=verbose)

                # make median combination of the de-rotated cube.
                tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=self.fwhm, delta_rot=delta_rot, full_output=False,
                                     verbose=verbose, nproc=self.nproc)
                tmp_tmp = mask_circle(tmp_tmp, mask_IWA_px)  # we mask the IWA
                write_fits(outpath_sub + 'final_ADI_simple.fits', tmp_tmp, verbose=verbose)

            ## SNR map
            if (not isfile(outpath_sub + 'final_ADI_simple_snrmap.fits') or overwrite) and do_snr_map:
                tmp = open_fits(outpath_sub + 'final_ADI_simple.fits', verbose=verbose)
                tmp_tmp = snrmap(tmp, self.fwhm, plot=False, nproc=self.nproc, verbose=debug)
                tmp_tmp = mask_circle(tmp_tmp, mask_IWA_px)
                write_fits(outpath_sub + 'final_ADI_simple_snrmap.fits', tmp_tmp, verbose=verbose)

            ## Contrast curve
            if (not isfile(outpath_sub + 'contrast_adi_madi.pdf') or overwrite) and do_adi_contrast:
                _ = contrast_curve(ADI_cube, derot_angles, psfn, self.fwhm, pxscale=self.pixel_scale, starphot=starphot,
                                   algo=median_sub, sigma=5, nbranch=nbranch, theta=0, inner_rad=mask_IWA,
                                   wedge=(0, 360), fc_snr=fc_snr, student=True, transmission=transmission, smooth=True,
                                   plot=True, dpi=300, debug=debug, save_plot=outpath_sub + 'contrast_simple_madi.pdf',
                                   verbose=verbose, nproc=self.nproc)
                plt.close('all')
            if verbose:
                print("======= Completed Median-ADI =======", flush=True)

        ################# Estimate sensitivity and inject planets ###################
        if fake_planet and not first_guess_skip:
            df_list = []
            PCA_ADI_cube = ADI_cube.copy()
            if verbose:
                print("======= Determining contrast to inject fake planets =======", flush=True)
                start_time = time_ini(verbose=False)
            # 3 sigma is to evaluate which contrast to inject the companion, final contrast curve is 5 sigma
            for nn, npc in enumerate(firstguess_pcs):
                pn_contr_curve_full_rr = contrast_curve(PCA_ADI_cube, derot_angles, psfn, self.fwhm, self.pixel_scale,
                                                        starphot=starphot, algo=pca, sigma=3, nbranch=nbranch, theta=0,
                                                        inner_rad=mask_IWA, wedge=(0, 360), fc_snr=fc_snr,
                                                        cube_ref=ref_cube, student=True, transmission=transmission,
                                                        plot=False, verbose=verbose, ncomp=int(npc),
                                                        svd_mode=svd_mode, nproc=self.nproc)
                df_list.append(pn_contr_curve_full_rr)
            pn_contr_curve_full_first_opt = pn_contr_curve_full_rr.copy()
            for jj in range(pn_contr_curve_full_first_opt.shape[0]):  # iterate over distance from centre
                sensitivities = []
                for nn, npc in enumerate(firstguess_pcs):  # iterate over tested principal components
                    sensitivities.append(df_list[nn]['sensitivity_student'][jj])  # sensitivity at that distance and npc
                print("Sensitivities at {} px: ".format(df_list[nn]['distance'][jj]), sensitivities, flush=True)
                idx_min = np.argmin(sensitivities)  # minimum sensitivity at that distance
                pn_contr_curve_full_first_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                pn_contr_curve_full_first_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                pn_contr_curve_full_first_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                pn_contr_curve_full_first_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                pn_contr_curve_full_first_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
            Df.to_csv(pn_contr_curve_full_first_opt,
                      path_or_buf=outpath_sub + 'TMP_first_guess_contrast_curve_PCA-ADI-full.csv', sep=',', na_rep='',
                      float_format=None)
            arr_dist = np.array(pn_contr_curve_full_first_opt['distance'])
            arr_contrast = np.array(pn_contr_curve_full_first_opt['sensitivity_student'])

            sensitivity_3sig_full_df = np.zeros(nfcp)
            for ff in range(nfcp):
                idx = find_nearest(arr_dist, rad_arr[ff])  # closest radial px separation from each companion to inject
                sensitivity_3sig_full_df[ff] = arr_contrast[idx]  # optimal contrast at that separation
            write_fits(outpath_sub + 'TMP_first_guess_3sig_sensitivity.fits', sensitivity_3sig_full_df, verbose=debug)

            # do this to inject the companions
            if verbose:
                timing(start_time)
                print("======= Injecting fake planets =======", flush=True)
                start_time = time_ini(verbose=False)
            for ns in range(nspi):  # iterate over PA separations to inject
                theta0 = ns * th_step  # changing PA (forms a spiral if there are >1 separations to inject)
                PCA_ADI_cube = ADI_cube.copy()  # refresh the cube so we don't accumulate fake companions
                for ff in range(nfcp):  # iterate over separations to inject
                    # inject the companions
                    flevel = np.median(starphot) * sensitivity_3sig_full_df[ff] * injection_fac
                    PCA_ADI_cube = cube_inject_companions(PCA_ADI_cube, psfn, derot_angles, flevel, self.pixel_scale,
                                                          rad_dists=rad_arr[ff:ff + 1], n_branches=1,
                                                          theta=(theta0 + ff * th_step) % 360,
                                                          imlib='opencv', verbose=verbose)
                write_fits(outpath_sub + 'PCA_cube_fcp_spi{:.0f}.fits'.format(ns), PCA_ADI_cube, verbose=debug)

            # if do_adi:
            #     sensitivity_5sig_adi_df = np.zeros(nfcp)
            if do_pca_full:
                id_npc_full_df = np.zeros(nfcp)
                sensitivity_5sig_full_df = np.zeros(nfcp)
            if do_pca_ann:
                id_npc_ann_df = np.zeros(nfcp)
                sensitivity_5sig_ann_df = np.zeros(nfcp)

            if verbose:
                timing(start_time)
                print("======= Injected fake planets =======", flush=True)

        ####################### PCA-ADI full ###########################
        if do_pca_full:
            test_pcs_str_list = [str(x) for x in test_pcs_full]
            ntest_pcs = len(test_pcs_full)
            test_pcs_str = "npc" + "-".join(test_pcs_str_list)

            # if there is no blob and if we haven't injected fake planets OR we want a full frame contrast curve
            if not fake_planet or (fake_planet and first_guess_skip):
                df_full_fgs = []
                PCA_ADI_cube = ADI_cube.copy()
                tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                if do_snr_map_opt:
                    tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                for nn, npc in enumerate(test_pcs_full):
                    if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite:
                        tmp_tmp[nn] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube,
                                          scale_list=None, ncomp=int(npc),
                                          svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                          source_xy=planet_pos, delta_rot=delta_rot, fwhm=self.fwhm, collapse='median',
                                          check_memory=True, full_output=False, verbose=verbose, nproc=self.nproc)
                    if (not isfile(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-full.csv') or overwrite) and \
                            (fake_planet and first_guess_skip):
                        contr_curve_full = contrast_curve(PCA_ADI_cube, derot_angles, psfn, self.fwhm,
                                                          self.pixel_scale,
                                                          starphot=starphot, algo=pca, sigma=5, nbranch=nbranch,
                                                          theta=0,
                                                          inner_rad=mask_IWA, wedge=(0, 360), fc_snr=fc_snr,
                                                          student=True,
                                                          transmission=transmission, plot=False, cube_ref=ref_cube,
                                                          scaling=None, verbose=verbose, ncomp=int(npc),
                                                          svd_mode=svd_mode,
                                                          nproc=self.nproc)
                        df_full_fgs.append(contr_curve_full)  # save data frame from each contrast curve
                    if (not isfile(
                            outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) \
                            and do_snr_map_opt:
                        tmp_tmp_tmp_tmp[nn] = pca(PCA_ADI_cube, angle_list=-derot_angles, cube_ref=ref_cube,
                                                  scale_list=None, ncomp=int(npc),
                                                  svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                                  source_xy=planet_pos, delta_rot=delta_rot, fwhm=self.fwhm,
                                                  collapse='median', check_memory=True,
                                                  full_output=False, verbose=verbose, nproc=self.nproc)
                if (not isfile(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-full.csv') or overwrite) and \
                        (fake_planet and first_guess_skip):
                    # get best sensitivity at each sampled distance by looping npcs
                    contr_curve_full_opt = contr_curve_full.copy()  # gets last data frame
                    for jj in range(contr_curve_full_opt.shape[0]):  # iterate over distances sampled from centre
                        sensitivities = []
                        for nn, npc in enumerate(test_pcs_full):  # iterate over tested principal components
                            sensitivities.append(df_full_fgs[nn]['sensitivity_student'][jj])  # sensitivity at that distance and npc
                        print("Sensitivities at {} px: ".format(df_full_fgs[nn]['distance'][jj]), sensitivities,
                              flush=True)
                        idx_min = np.argmin(sensitivities)  # minimum sensitivity at that distance
                        contr_curve_full_opt['sensitivity_student'][jj] = df_full_fgs[idx_min]['sensitivity_student'][jj]
                        contr_curve_full_opt['sensitivity_gaussian'][jj] = df_full_fgs[idx_min]['sensitivity_gaussian'][jj]
                        contr_curve_full_opt['throughput'][jj] = df_full_fgs[idx_min]['throughput'][jj]
                        contr_curve_full_opt['noise'][jj] = df_full_fgs[idx_min]['noise'][jj]
                        contr_curve_full_opt['sigma corr'][jj] = df_full_fgs[idx_min]['sigma corr'][jj]
                    Df.to_csv(contr_curve_full_opt,
                              path_or_buf=outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-full.csv', sep=',',
                              na_rep='', float_format=None)

                if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite:
                    write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', tmp_tmp, verbose=debug)

                if (not isfile(
                        outpath_sub + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits') or overwrite) and do_snr_map_opt:
                    write_fits(outpath_sub + 'neg_PCA-ADI_full_' + test_pcs_str + '.fits',
                               tmp_tmp_tmp_tmp, verbose=debug)
                ### Convolution
                if not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits') or overwrite:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=debug)
                    for nn in range(tmp.shape[0]):
                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss', fwhm_size=self.fwhm, conv_mode='convfft')
                    write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_conv.fits', tmp, verbose=debug)

                ### SNR map
                if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits') or overwrite) \
                        and do_snr_map:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=debug)
                    for pp in range(ntest_pcs):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, nproc=self.nproc, verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap.fits', tmp, verbose=debug)

                ### SNR map optimized
                if (not isfile(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) \
                        and do_snr_map_opt:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '.fits', verbose=verbose)
                    for pp in range(ntest_pcs):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, array2=tmp_tmp_tmp_tmp[pp], incl_neg_lobes=False,
                                         plot=False, nproc=self.nproc, verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_full_' + test_pcs_str + '_snrmap_opt.fits', tmp,
                               verbose=verbose)

            elif fake_planet and not first_guess_skip:  # use the cubes with fake companions injected
                snr_tmp = np.zeros([nspi, ntest_pcs, nfcp])
                tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                for ns in range(nspi):
                    theta0 = ns * th_step
                    PCA_ADI_cube = open_fits(outpath_sub + 'PCA_cube_fcp_spi{:.0f}.fits'.format(ns), verbose=debug)
                    for pp, npc in enumerate(test_pcs_full):
                        tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None,
                                          mask_center_px=mask_IWA_px, ncomp=int(npc), scaling=None, fwhm=self.fwhm,
                                          collapse='median', full_output=False, verbose=verbose, nproc=self.nproc,
                                          svd_mode=svd_mode)
                        for ff in range(nfcp):  # determine SNR for each companion
                            xx_fcp = cx + rad_arr[ff] * np.cos(np.deg2rad(theta0 + ff * th_step))
                            yy_fcp = cy + rad_arr[ff] * np.sin(np.deg2rad(theta0 + ff * th_step))
                            snr_tmp[ns, pp, ff] = snr(tmp_tmp[pp], (xx_fcp, yy_fcp), self.fwhm, plot=False,
                                                      exclude_negative_lobes=True, verbose=verbose)
                    write_fits(outpath_sub+'TMP_PCA-ADI_full_'+test_pcs_str+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp,
                               verbose=debug)
                snr_fcp = np.median(snr_tmp, axis=0)
                write_fits(outpath_sub+'final_PCA-ADI_full_SNR_fcps_'+test_pcs_str+'.fits', snr_fcp, verbose=debug)

                # Find best npc for each radius
                for ff in range(nfcp):
                    idx_best_snr = np.argmax(snr_fcp[:, ff])
                    id_npc_full_df[ff] = test_pcs_full[idx_best_snr]

                # Re-run at optimal npcs
                PCA_ADI_cube = ADI_cube.copy()
                tmp_tmp = np.zeros([nfcp, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                test_pcs_str_list = [str(int(x)) for x in id_npc_full_df]
                test_pcs_str = "npc_opt" + "-".join(test_pcs_str_list)
                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr * self.pixel_scale]
                test_rad_str = "rad" + "-".join(test_rad_str_list)
                for pp, npc in enumerate(id_npc_full_df):
                    tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None,
                                      ncomp=int(npc), svd_mode=svd_mode, scaling=None, mask_center_px=mask_IWA_px,
                                      fwhm=self.fwhm, collapse='median', full_output=False, verbose=verbose)
                write_fits(outpath_sub + 'final_PCA-ADI_full_{}_at_{}as.fits'.format(test_pcs_str, test_rad_str),
                           tmp_tmp, verbose=debug)
                write_fits(outpath_sub + 'final_PCA-ADI_full_npc_id_at_{}as.fits'.format(test_rad_str), id_npc_full_df,
                           verbose=debug)
                ### Convolution
                if not isfile(outpath_sub + 'final_PCA-ADI_full_{}_at_{}as_conv.fits'.format(
                        test_pcs_str, test_rad_str)) or overwrite:
                    tmp = open_fits(
                        outpath_sub + 'final_PCA-ADI_full_{}_at_{}as.fits'.format(test_pcs_str, test_rad_str),
                        verbose=debug)
                    for nn in range(tmp.shape[0]):
                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss', fwhm_size=self.fwhm, conv_mode='convfft')
                    write_fits(
                        outpath_sub + 'final_PCA-ADI_full_{}_at_{}as_conv.fits'.format(test_pcs_str, test_rad_str)
                        , tmp, verbose=debug)
                ### SNR map
                if (not isfile(outpath_sub + 'final_PCA-ADI_full_{}_at_{}as_snrmap.fits'.format(
                        test_pcs_str, test_rad_str)) or overwrite) and do_snr_map:
                    tmp = open_fits(
                        outpath_sub + 'final_PCA-ADI_full_{}_at_{}as.fits'.format(test_pcs_str, test_rad_str)
                        , verbose=debug)
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, nproc=self.nproc, verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_full_{}_at_{}as'.format(test_pcs_str, test_rad_str) +
                               '_snrmap.fits', tmp, verbose=debug)

            if verbose:
                print("======= Completed PCA Full Frame =======", flush=True)

        ######################## PCA-ADI annular #######################
        if do_pca_ann:
            test_pcs_str_list = [str(x) for x in test_pcs_ann]
            ntest_pcs = len(test_pcs_ann)
            test_pcs_str = "npc" + "-".join(test_pcs_str_list)

            if not fake_planet or (fake_planet and first_guess_skip):
                df_ann_fgs = []
                PCA_ADI_cube = ADI_cube.copy()
                tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                if debug:
                    array_der = np.zeros([ntest_pcs, PCA_ADI_cube.shape[0], PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                    array_out = np.zeros([ntest_pcs, PCA_ADI_cube.shape[0], PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                if do_snr_map_opt:
                    tmp_tmp_tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])

                for nn, npc in enumerate(test_pcs_ann):
                    if debug and ((not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits') and
                                   not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits'))
                                  or overwrite):
                        # saves residuals and median if debug is true and they either dont exist or are to be overwritten
                        array_out[nn], array_der[nn], tmp_tmp[nn] = pca_annular(PCA_ADI_cube, derot_angles,
                                                                                cube_ref=ref_cube, scale_list=None,
                                                                                radius_int=mask_IWA_px, fwhm=self.fwhm,
                                                                                asize=ann_sz * self.fwhm, n_segments=1,
                                                                                delta_rot=delta_rot, ncomp=int(npc),
                                                                                svd_mode=svd_mode, nproc=self.nproc,
                                                                                min_frames_lib=max(npc, 10),
                                                                                max_frames_lib=200, tol=1e-1,
                                                                                scaling=None, imlib='opencv',
                                                                                interpolation='lanczos4',
                                                                                collapse='median',
                                                                                ifs_collapse_range='all',
                                                                                full_output=debug, verbose=verbose)
                    else:
                        if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits') or overwrite:
                            tmp_tmp[nn] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                                      radius_int=mask_IWA_px, fwhm=self.fwhm, asize=ann_sz * self.fwhm,
                                                      n_segments=1, delta_rot=delta_rot, ncomp=int(npc),
                                                      svd_mode=svd_mode, nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                      max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                      interpolation='lanczos4', collapse='median',
                                                      ifs_collapse_range='all', full_output=False, verbose=verbose)

                    if (not isfile(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-ann.csv') or overwrite) and \
                            (fake_planet and first_guess_skip):
                        contr_curve_ann = contrast_curve(PCA_ADI_cube, derot_angles, psfn, self.fwhm,
                                                         self.pixel_scale, starphot=starphot, algo=pca_annular,
                                                         sigma=5, nbranch=nbranch, theta=0, inner_rad=mask_IWA,
                                                         wedge=(0, 360), fc_snr=fc_snr, student=True,
                                                         transmission=transmission, plot=False,
                                                         verbose=verbose, ncomp=int(npc),
                                                         svd_mode=svd_mode, radius_int=mask_IWA_px,
                                                         asize=ann_sz * self.fwhm, delta_rot=delta_rot,
                                                         cube_ref=ref_cube, scaling=None,
                                                         min_frames_lib=max(int(npc), 10),
                                                         max_frames_lib=200, nproc=self.nproc)
                        df_ann_fgs.append(contr_curve_ann)  # save data frame from each contrast curve

                    if (not isfile(
                            outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                        tmp_tmp_tmp_tmp[nn] = pca_annular(PCA_ADI_cube, -derot_angles, cube_ref=ref_cube,
                                                          scale_list=None, radius_int=mask_IWA_px, fwhm=self.fwhm,
                                                          asize=ann_sz * self.fwhm, n_segments=1, delta_rot=delta_rot,
                                                          ncomp=int(npc), svd_mode=svd_mode,
                                                          nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                          max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                          interpolation='lanczos4', collapse='median',
                                                          ifs_collapse_range='all', full_output=False, verbose=verbose)

                if (not isfile(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-ann.csv') or overwrite) and \
                        (fake_planet and first_guess_skip):
                    # similar loop to before, get best sensitivity at each distance sampled by looping npcs
                    contr_curve_ann_opt = contr_curve_ann.copy()  # gets last data frame
                    for jj in range(contr_curve_ann_opt.shape[0]):  # iterate over distances sampled from centre
                        sensitivities = []
                        for nn, npc in enumerate(test_pcs_ann):  # iterate over tested principal components
                            sensitivities.append(
                                df_ann_fgs[nn]['sensitivity_student'][jj])  # sensitivity at that distance and npc
                        print("Sensitivities at {} px: ".format(df_ann_fgs[nn]['distance'][jj]), sensitivities,
                              flush=True)
                        idx_min = np.argmin(sensitivities)  # minimum sensitivity at that distance
                        contr_curve_ann_opt['sensitivity_student'][jj] = df_ann_fgs[idx_min]['sensitivity_student'][jj]
                        contr_curve_ann_opt['sensitivity_gaussian'][jj] = df_ann_fgs[idx_min]['sensitivity_gaussian'][jj]
                        contr_curve_ann_opt['throughput'][jj] = df_ann_fgs[idx_min]['throughput'][jj]
                        contr_curve_ann_opt['noise'][jj] = df_ann_fgs[idx_min]['noise'][jj]
                        contr_curve_ann_opt['sigma corr'][jj] = df_ann_fgs[idx_min]['sigma corr'][jj]
                    Df.to_csv(contr_curve_ann_opt,
                              path_or_buf=outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-ann.csv', sep=',',
                              na_rep='', float_format=None)

                if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits') or overwrite:
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp, verbose=verbose)
                if debug and ((not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits') and
                               not isfile(
                                   outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits')) or overwrite):
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals.fits', array_out,
                               verbose=debug)
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_residuals-derot.fits', array_der,
                               verbose=debug)

                if (not isfile(
                        outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                    write_fits(outpath_sub + 'neg_PCA-ADI_ann_' + test_pcs_str + '.fits', tmp_tmp_tmp_tmp, verbose=debug)

                ### Convolution
                if not isfile(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits') or overwrite:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=debug)
                    for nn in range(tmp.shape[0]):
                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss', fwhm_size=self.fwhm, conv_mode='convfft')
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_conv.fits', tmp, verbose=debug)

                ### SNR map
                if (not isfile(
                        outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap.fits') or overwrite) and do_snr_map:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=debug)
                    for pp in range(ntest_pcs):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, nproc=self.nproc, verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap.fits', tmp, verbose=debug)
                ### SNR map optimized
                if (not isfile(
                        outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits') or overwrite) and do_snr_map_opt:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '.fits', verbose=debug)
                    for pp in range(ntest_pcs):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, array2=tmp_tmp_tmp_tmp[pp], nproc=self.nproc,
                                         verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_' + test_pcs_str + '_snrmap_opt.fits', tmp, verbose=debug)

            elif fake_planet and not first_guess_skip:
                snr_tmp_tmp = np.zeros([nspi, ntest_pcs, nfcp])
                tmp_tmp = np.zeros([ntest_pcs, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                for ns in range(nspi):
                    theta0 = ns * th_step
                    PCA_ADI_cube = open_fits(outpath_sub + 'PCA_cube_fcp_spi{:.0f}.fits'.format(ns), verbose=debug)
                    for pp, npc in enumerate(test_pcs_ann):
                        tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, scale_list=None,
                                                  radius_int=mask_IWA_px, fwhm=self.fwhm, asize=ann_sz * self.fwhm,
                                                  n_segments=1, delta_rot=delta_rot, ncomp=int(npc),
                                                  svd_mode=svd_mode, nproc=self.nproc, min_frames_lib=max(npc, 10),
                                                  max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                                                  interpolation='lanczos4', collapse='median', ifs_collapse_range='all',
                                                  full_output=False, verbose=verbose)
                        for ff in range(nfcp):
                            xx_fcp = cx + rad_arr[ff] * np.cos(np.deg2rad(theta0 + ff * th_step))
                            yy_fcp = cy + rad_arr[ff] * np.sin(np.deg2rad(theta0 + ff * th_step))
                            snr_tmp_tmp[ns, pp, ff] = snr(tmp_tmp[pp], (xx_fcp, yy_fcp), self.fwhm, plot=False,
                                                          exclude_negative_lobes=True, verbose=verbose)
                    write_fits(outpath_sub+'TMP_PCA-ADI_ann_'+test_pcs_str+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp,
                               verbose=debug)
                snr_fcp = np.median(snr_tmp_tmp, axis=0)
                write_fits(outpath_sub + 'final_PCA-ADI_ann_SNR_fcps_' + test_pcs_str + '.fits', snr_fcp, verbose=debug)

                # Find best npc for each radius
                for ff in range(nfcp):
                    idx_best_snr = np.argmax(snr_fcp[:, ff])
                    id_npc_ann_df[ff] = test_pcs_ann[idx_best_snr]

                # Final PCA-ADI annular with optimal npcs
                PCA_ADI_cube = ADI_cube.copy()
                tmp_tmp = np.zeros([nfcp, PCA_ADI_cube.shape[1], PCA_ADI_cube.shape[2]])
                test_pcs_str_list = [str(int(x)) for x in id_npc_ann_df]
                test_pcs_str = "npc_opt" + "-".join(test_pcs_str_list)
                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr * self.pixel_scale]
                test_rad_str = "rad" + "-".join(test_rad_str_list)
                for pp, npc in enumerate(id_npc_ann_df):
                    tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, radius_int=mask_IWA_px, fwhm=self.fwhm,
                                              asize=ann_sz * self.fwhm, delta_rot=delta_rot, ncomp=int(npc),
                                              svd_mode=svd_mode, max_frames_lib=200, cube_ref=ref_cube, scaling=None,
                                              min_frames_lib=max(npc, 10), collapse='median', full_output=False,
                                              verbose=verbose, nproc=self.nproc)
                write_fits(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as'.format(test_pcs_str, test_rad_str) + '.fits',
                           tmp_tmp, verbose=debug)
                write_fits(outpath_sub+'final_PCA-ADI_ann_npc_id_at_{}as'.format(test_rad_str)+'.fits', id_npc_ann_df,
                           verbose=debug)

                ### Convolution
                if not isfile(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as'.format(test_pcs_str, test_rad_str)
                              + '_conv.fits') or overwrite:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as'.format(test_pcs_str, test_rad_str) +
                                    '.fits', verbose=debug)
                    for nn in range(tmp.shape[0]):
                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss', fwhm_size=self.fwhm, conv_mode='convfft')
                    write_fits(
                        outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as'.format(test_pcs_str, test_rad_str) + '_conv.fits',
                        tmp, verbose=debug)
                ### SNR map
                if (not isfile(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as_snrmap.fits'.format(
                        test_pcs_str, test_rad_str)) or overwrite) and do_snr_map:
                    tmp = open_fits(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as.fits'.format(test_pcs_str, test_rad_str)
                                    , verbose=debug)
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], self.fwhm, plot=False, nproc=self.nproc, verbose=debug)
                    tmp = mask_circle(tmp, mask_IWA_px)
                    write_fits(outpath_sub + 'final_PCA-ADI_ann_{}_at_{}as'.format(test_pcs_str, test_rad_str) +
                               '_snrmap.fits', tmp, verbose=debug)

            if verbose:
                print("======= Completed PCA Annular =======", flush=True)

        # Final 5 sigma contrast curve with optimal npcs
        if fake_planet and not first_guess_skip:
            PCA_ADI_cube = ADI_cube.copy()
            if verbose:
                print("======= Creating optimal contrast curve =======", flush=True)
            if do_pca_full:
                df_list = []
                for rr, rad in enumerate(rad_arr):  # contrast at each separation sampled
                    pn_contr_curve_full_rr = contrast_curve(PCA_ADI_cube, derot_angles, psfn, self.fwhm,
                                                            self.pixel_scale, starphot=starphot, algo=pca, sigma=5,
                                                            nbranch=nbranch, theta=0, inner_rad=mask_IWA, wedge=(0,360),
                                                            fc_snr=fc_snr, student=True, transmission=transmission,
                                                            plot=False, cube_ref=ref_cube, scaling=None,
                                                            verbose=verbose, ncomp=int(id_npc_full_df[rr]),
                                                            svd_mode=svd_mode, nproc=self.nproc)
                    Df.to_csv(pn_contr_curve_full_rr,
                              path_or_buf=outpath_sub + 'contrast_curve_PCA-ADI-full_optimal_at_{:.1f}as.csv'.format(
                                  rad * self.pixel_scale),
                              sep=',', na_rep='', float_format=None)
                    df_list.append(pn_contr_curve_full_rr)
                pn_contr_curve_full_opt = pn_contr_curve_full_rr.copy()
                for jj in range(pn_contr_curve_full_opt.shape[0]):  # loop distances
                    sensitivities = []
                    for rr, rad in enumerate(rad_arr):
                        sensitivities.append(
                            df_list[rr]['sensitivity_student'][jj])  # sensitivity at sampled separation
                    print("Sensitivities at {} px: ".format(df_list[rr]['distance'][jj]), sensitivities, flush=True)
                    idx_min = np.argmin(sensitivities)  # best sensitivity out of all sampled separations
                    pn_contr_curve_full_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                    pn_contr_curve_full_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                    pn_contr_curve_full_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                    pn_contr_curve_full_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                    pn_contr_curve_full_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                Df.to_csv(pn_contr_curve_full_opt,
                          path_or_buf=outpath_sub + 'final_optimal_contrast_curve_PCA-ADI-full.csv', sep=',', na_rep='',
                          float_format=None)
                arr_dist = np.array(pn_contr_curve_full_opt['distance'])
                arr_contrast = np.array(pn_contr_curve_full_opt['sensitivity_student'])
                for ff in range(nfcp):
                    idx = find_nearest(arr_dist, rad_arr[ff])
                    sensitivity_5sig_full_df[ff] = arr_contrast[idx]

            if do_pca_ann:
                PCA_ADI_cube = ADI_cube.copy()  # ensure a fresh cube
                df_list = []
                for rr, rad in enumerate(rad_arr):
                    pn_contr_curve_ann_rr = contrast_curve(PCA_ADI_cube, derot_angles, psfn, self.fwhm,
                                                           self.pixel_scale, starphot=starphot, algo=pca_annular,
                                                           sigma=5, nbranch=nbranch, theta=0, inner_rad=mask_IWA,
                                                           wedge=(0, 360), fc_snr=fc_snr, student=True,
                                                           transmission=transmission, plot=False,
                                                           verbose=verbose, ncomp=int(id_npc_ann_df[rr]),
                                                           svd_mode=svd_mode, radius_int=mask_IWA_px,
                                                           asize=ann_sz * self.fwhm, delta_rot=delta_rot,
                                                           cube_ref=ref_cube, scaling=None,
                                                           min_frames_lib=max(id_npc_ann_df[rr], 10),
                                                           max_frames_lib=200, nproc=self.nproc)
                    Df.to_csv(pn_contr_curve_ann_rr,
                              path_or_buf=outpath_sub + 'contrast_curve_PCA-ADI-ann_optimal_at_{:.1f}as.csv'.format(
                                  rad * self.pixel_scale),
                              sep=',', na_rep='', float_format=None)
                    df_list.append(pn_contr_curve_ann_rr)
                pn_contr_curve_ann_opt = pn_contr_curve_ann_rr.copy()

                for jj in range(pn_contr_curve_ann_opt.shape[0]):
                    sensitivities = []
                    for rr, rad in enumerate(rad_arr):
                        sensitivities.append(df_list[rr]['sensitivity_student'][jj])
                    print("Sensitivities at {} px: ".format(df_list[rr]['distance'][jj]), sensitivities, flush=True)
                    idx_min = np.argmin(sensitivities)  # absolute best overall principal component
                    pn_contr_curve_ann_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                    pn_contr_curve_ann_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                    pn_contr_curve_ann_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                    pn_contr_curve_ann_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                    pn_contr_curve_ann_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                Df.to_csv(pn_contr_curve_ann_opt,
                          path_or_buf=outpath_sub + 'final_optimal_contrast_curve_PCA-ADI-ann.csv',
                          sep=',', na_rep='', float_format=None)
                arr_dist = np.array(pn_contr_curve_ann_opt['distance'])
                arr_contrast = np.array(pn_contr_curve_ann_opt['sensitivity_student'])
                for ff in range(nfcp):
                    idx = find_nearest(arr_dist, rad_arr[ff])
                    sensitivity_5sig_ann_df[ff] = arr_contrast[idx]

            # plot final contrast curve
            plt.close('all')
            plt.figure(dpi=300)
            plt.title('5\u03C3 contrast curve for {} {}'.format(self.source, self.details))
            plt.ylabel('Contrast')
            plt.xlabel('Separation ["]')
            # if do_adi:
            #     plt.semilogy(pn_contr_curve_adi['distance'] * plsc, pn_contr_curve_adi['sensitivity_student'],
            #                  'r', linewidth=2, label='median-ADI (Student correction)')
            if do_pca_full:
                plt.semilogy(pn_contr_curve_full_opt['distance'] * self.pixel_scale,
                             pn_contr_curve_full_opt['sensitivity_student'], 'b', linewidth=2,
                             label='PCA-ADI full frame (Student, optimal)')

            if do_pca_ann:
                plt.semilogy(pn_contr_curve_ann_opt['distance'] * self.pixel_scale,
                             pn_contr_curve_ann_opt['sensitivity_student'], 'r', linewidth=2,
                             label='PCA-ADI annular (Student, optimal)')
            plt.legend()
            try:
                plt.savefig(outpath_sub + 'contr_curves.pdf', format='pdf')
            except:
                pass

            plt.close('all')
            plt.figure(dpi=300)
            plt.title('5\u03C3 contrast curve for {} {}'.format(self.source, self.details))
            plt.ylabel('Contrast [mag]')
            plt.gca().invert_yaxis()
            plt.xlabel('Separation ["]')
            # if do_adi:
            #     plt.plot(pn_contr_curve_adi['distance'] * plsc,
            #              -2.5 * np.log10(pn_contr_curve_adi['sensitivity_student']), 'r', linewidth=2,
            #              label='median-ADI (Student correction)')
            if do_pca_full:
                plt.plot(pn_contr_curve_full_opt['distance'] * self.pixel_scale,
                         -2.5 * np.log10(pn_contr_curve_full_opt['sensitivity_student']), 'b', linewidth=2,
                         label='PCA-ADI full frame (Student, optimal)')

            if do_pca_ann:
                plt.plot(pn_contr_curve_ann_opt['distance'] * self.pixel_scale,
                         -2.5 * np.log10(pn_contr_curve_ann_opt['sensitivity_student']), 'r', linewidth=2,
                         label='PCA-ADI annular (Student, optimal)')
            plt.legend()
            try:
                plt.savefig(outpath_sub + 'contr_curves_app_magnitude.pdf', format='pdf')
            except:
                pass
            plt.close('all')

        elif fake_planet and first_guess_skip:  # no fake planets injected, just plot the best contrast at each radii
            if do_pca_full:
                contr_full_df = read_csv(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-full.csv')
            if do_pca_ann:
                contr_ann_df = read_csv(outpath_sub + 'final_skip-fcp_contrast_curve_PCA-ADI-ann.csv')

            plt.close('all')
            plt.figure(dpi=300)
            plt.title('5\u03C3 contrast curve for {} {}'.format(self.source, self.details))
            plt.ylabel('Contrast [mag]')
            plt.gca().invert_yaxis()
            plt.xlabel('Separation ["]')
            if do_pca_full:
                plt.plot(contr_full_df['distance'] * self.pixel_scale,
                         -2.5 * np.log10(contr_full_df['sensitivity_student']),
                         'b', linewidth=2, label='PCA-ADI full frame (Student)')
            if do_pca_ann:
                plt.plot(contr_ann_df['distance'] * self.pixel_scale,
                         -2.5 * np.log10(contr_ann_df['sensitivity_student']),
                         'r', linewidth=2, label='PCA-ADI annular (Student)')
            plt.legend()
            try:
                plt.savefig(outpath_sub + 'contr_curve_skip-fcp.pdf', format='pdf')
            except:
                pass
            plt.close('all')

        if verbose:
            print("======= Finished post-processing =======", flush=True)

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
        ----------
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

        print("======= Starting NEGFC....=======", flush=True)
        if guess_xy is None and do_firstguess is True:
            raise ValueError("Enter an approximate location into guess_xy!")

        if weights is True and coronagraph is True:
            raise ValueError("Dataset cannot be both non-coronagraphic and coronagraphic!!")

        outpath_sub = self.outpath + "negfc/"

        if not isdir(outpath_sub):
            os.system("mkdir " + outpath_sub)

        if verbose:
            print('Input path is {}'.format(self.inpath))
            print('Output path is {}'.format(outpath_sub), flush=True)

        source = self.dataset_dict['source']
        tn_shift = 0.572  # ± 0.178, Launhardt et al. 2020, true North offset for NACO

        ADI_cube_name = '{}_master_cube.fits'  # template name for input master cube
        derot_ang_name = 'derot_angles.fits'  # template name for corresponding input derotation angles
        psfn_name = "master_unsat_psf_norm.fits"  # normalised PSF

        ADI_cube = open_fits(self.inpath + ADI_cube_name.format(source), verbose=verbose)
        derot_angles = open_fits(self.inpath + derot_ang_name, verbose=verbose) + tn_shift
        psfn = open_fits(self.inpath + psfn_name, verbose=verbose)
        ref_cube = None
        svd_mode = 'lapack'

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
        f_range = np.geomspace(0.1, 201, 1000)
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
            # data provided by Valentin Christiaens. First entry in both columns was not zero, but VIP adds it anyway
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

        if (not isfile(outpath_sub + label_pca + "_npc{}_simplex_results.fits".format(
                opt_npc)) or overwrite) and do_firstguess:
            # find r, theta based on the provided estimate location
            cy, cx = frame_center(ADI_cube[0])
            dy_pl = guess_xy[0][1] - cy
            dx_pl = guess_xy[0][0] - cx
            r_pl = np.sqrt(np.power(dx_pl, 2) + np.power(dy_pl, 2))  # pixel distance to the guess location
            theta_pl = (np.rad2deg(np.arctan2(dy_pl, dx_pl))) % 360  # theta (angle) to the guess location
            print("Estimated (r, theta) before first guess = ({:.1f},{:.1f})".format(r_pl, theta_pl), flush=True)

            ini_state = firstguess(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                   planets_xy_coord=guess_xy, fwhm=self.fwhm,
                                   annulus_width=12, aperture_radius=ap_rad, cube_ref=ref_cube,
                                   svd_mode=svd_mode, scaling=None, fmerit='stddev', imlib='opencv',
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

            delta_theta_min = np.rad2deg(np.arctan(4. / r_pl))  # at least the angle corresponding to 2 azimuthal pixels
            delta_theta = max(delta_theta_min, 5.)
            bounds = [(max(r_pl - asize / 2., 1), r_pl + asize / 2.),  # radius
                      (theta_pl - delta_theta, theta_pl + delta_theta),  # angle
                      (0, 5 * abs(ini_state[2]))]

            if ini_state[0] < bounds[0][0] or ini_state[0] > bounds[0][1] or ini_state[1] < bounds[1][0] or \
                    ini_state[1] > bounds[1][1] or ini_state[2] < bounds[2][0] or ini_state[2] > bounds[2][1]:
                print("!!! WARNING: simplex results not in original bounds - NEGFC simplex MIGHT HAVE FAILED !!!", flush=True)
                ini_state = np.array([r_pl, theta_pl, abs(ini_state[2])])

            if verbose is True:
                verbosity = 2
                print('MCMC NEGFC sampling is about to begin...', flush=True)
            else:
                verbosity = 0

            final_chain = mcmc_negfc_sampling(ADI_cube, derot_angles, psfn, ncomp=opt_npc, plsc=self.pixel_scale,
                                              initial_state=ini_state, fwhm=self.fwhm, weights=weights,
                                              annulus_width=12, aperture_radius=ap_rad, cube_ref=ref_cube,
                                              svd_mode=svd_mode, scaling=None, fmerit='stddev',
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
            isamples_flat = final_chain[:, int(final_chain.shape[1] // (1 / 0.3)):, :].reshape(
                (-1, 3))  # 0.3 is the burnin
            vals, err = confidence(isamples_flat, cfd=68.27, bins=100, gaussian_fit=False, weights=weights,
                                   verbose=verbose, save=True, output_dir=outpath_sub, filename='confidence.txt',
                                   plsc=self.pixel_scale)

            labels = ['r', 'theta', 'f']
            mcmc_res = np.zeros([3, 3])
            # pull the values and confidence interval out for saving
            for i in range(3):
                mcmc_res[i, 0] = vals[labels[i]]
                mcmc_res[i, 1] = err[labels[i]][0]
                mcmc_res[i, 2] = err[labels[i]][1]
            write_fits(outpath_sub + 'mcmc_results.fits', mcmc_res)

            # now gaussian fit
            gvals, gerr = confidence(isamples_flat, cfd=68.27, bins=100, gaussian_fit=True, weights=weights,
                                     verbose=verbose, save=True, output_dir=outpath_sub,
                                     filename='confidence_gauss.txt',
                                     plsc=self.pixel_scale)

            mcmc_res = np.zeros([3, 2])
            for i in range(3):
                mcmc_res[i, 0] = gvals[i]
                mcmc_res[i, 1] = gerr[i]
            write_fits(outpath_sub + 'mcmc_results_gauss.fits', mcmc_res)

        if inject_neg:
            pca_res = np.zeros([ADI_cube.shape[1], ADI_cube.shape[2]])
            pca_res_emp = pca_res.copy()
            planet_params = open_fits(outpath_sub + 'mcmc_results.fits')
            flux_psf_name = "master_unsat-stellarpsf_fluxes.fits"
            star_flux = open_fits(self.inpath + flux_psf_name, verbose=verbose)[1]

            ADI_cube_emp = cube_inject_companions(ADI_cube, psfn, derot_angles,
                                                  flevel=-planet_params[2, 0] * star_flux, plsc=self.pixel_scale,
                                                  rad_dists=[planet_params[0, 0]],
                                                  n_branches=1, theta=planet_params[1, 0],
                                                  imlib='opencv', interpolation='lanczos4',
                                                  verbose=verbose, transmission=transmission)
            write_fits(outpath_sub + 'ADI_cube_empty.fits', ADI_cube_emp)  # the cube with the negative flux injected

            if algo == pca_annular:
                radius_int = int(np.floor(r_pl - asize / 2))  # asize is 3 * FWHM, rounds down. To skip the inner region
                # crop the cube to just larger than the annulus to improve the speed of PCA
                crop_sz = int(2 * np.ceil(r_pl + asize + 1))  # rounds up
                if not crop_sz % 2:  # make sure the crop is odd
                    crop_sz += 1
                if crop_sz < ADI_cube.shape[1] and crop_sz < ADI_cube.shape[2]:  # crop if crop_sz is smaller than cube
                    pad = int((ADI_cube.shape[1] - crop_sz) / 2)
                    crop_cube = cube_crop_frames(ADI_cube, crop_sz, verbose=verbose)
                else:
                    crop_cube = ADI_cube  # dont crop if the cube is already smaller

                pca_res_tmp = pca_annular(crop_cube, derot_angles, cube_ref=ref_cube, radius_int=radius_int,
                                          fwhm=self.fwhm, asize=asize, delta_rot=delta_rot, ncomp=opt_npc,
                                          svd_mode=svd_mode, scaling=None, imlib='opencv', interpolation='lanczos4',
                                          nproc=self.nproc, min_frames_lib=max(opt_npc, 10), verbose=verbose,
                                          full_output=False)

                pca_res = np.pad(pca_res_tmp, pad, mode='constant', constant_values=0)
                write_fits(outpath_sub + 'pca_annular_res_npc{}.fits'.format(opt_npc), pca_res)

                # emp
                if crop_sz < ADI_cube_emp.shape[1] and crop_sz < ADI_cube_emp.shape[2]:
                    pad = int((ADI_cube_emp.shape[1] - crop_sz) / 2)
                    crop_cube = cube_crop_frames(ADI_cube_emp, crop_sz, verbose=verbose)
                else:
                    crop_cube = ADI_cube_emp
                del ADI_cube_emp
                del ADI_cube

                pca_res_tmp = pca_annular(crop_cube, derot_angles, cube_ref=ref_cube, radius_int=radius_int,
                                          fwhm=self.fwhm, asize=asize, delta_rot=delta_rot, ncomp=opt_npc,
                                          svd_mode=svd_mode, scaling=None, imlib='opencv', interpolation='lanczos4',
                                          nproc=self.nproc, min_frames_lib=max(opt_npc, 10), verbose=verbose,
                                          full_output=False)

                # pad again now
                pca_res_emp = np.pad(pca_res_tmp, pad, mode='constant', constant_values=0)
                write_fits(outpath_sub + 'pca_annular_res_empty_npc{}.fits'.format(opt_npc), pca_res_emp)
