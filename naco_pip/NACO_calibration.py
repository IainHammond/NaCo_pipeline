#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applies necessary calibration to the cubes and corrects NACO biases

@author: lewis, iain
"""
__author__ = 'Lewis Picker, Iain Hammond'
__all__ = ['raw_dataset', 'find_filtered_max']

from pdb import set_trace
from os import makedirs, system
from os.path import isdir, isfile

import matplotlib
import numpy as np
import pyprind
from matplotlib import pyplot as plt
from photutils import CircularAperture, aperture_photometry
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation

from hciplot import plot_frames
from vip_hci.config import time_ini, timing
from vip_hci.fits import open_fits, write_fits
from vip_hci.fm import normalize_psf, find_nearest
from vip_hci.metrics import detection
from vip_hci.preproc import frame_crop, cube_crop_frames, frame_shift, \
    cube_subtract_sky_pca, cube_correct_nan, cube_fix_badpix_isolated, cube_fix_badpix_clump, \
    cube_recenter_2dfit, frame_fix_badpix_isolated, cube_detect_badfr_pxstats, approx_stellar_position
from vip_hci.var import frame_center, get_annulus_segments, frame_filter_lowpass, \
    mask_circle, dist, fit_2dgaussian, frame_filter_highpass, get_circle
from naco_pip.NACO_classification import find_AGPM

matplotlib.use('Agg')  # show option for plot is unavailable with this option

def find_shadow_list(self, file_list, threshold=None, plot=True, verbose=True, debug=False):
    """
    In coronographic NACO data there is a Lyot stop causing an outer shadow on the detector.
    This function will return the radius and central position of the circular shadow.
    """

    cube = open_fits(self.inpath + file_list[0], verbose=debug)
    ny, nx = cube.shape[-2:]
    if cube.ndim == 3:
        median_frame = cube[-1]
    else:
        median_frame = cube.copy()
    median_frame = frame_filter_lowpass(median_frame, median_size=7, mode='median')
    median_frame = frame_filter_lowpass(median_frame, mode='gauss', fwhm_size=5)
    yx = find_AGPM(self.inpath + file_list[0], verbose=False, debug=debug)
    ycom = yx[0]
    xcom = yx[1]
    write_fits(self.outpath + 'shadow_median_frame.fits', median_frame, verbose=debug)

    # create similar shadow centred at the origin
    if threshold is None:
        shadow = np.where(median_frame < np.nanmedian(median_frame), 1, 0)
        #shadow = np.where(median_frame < np.percentile(median_frame, 45), 1, 0)
        # shadow[:, :10] = 1
        # shadow[:10, :] = 1
        # shadow[:, -10:] = 1
        # shadow[-10:, :] = 1
    else:
        shadow = np.where(median_frame > threshold, 1, 0)  # lyot shadow
    area = sum(sum(shadow))
    r = np.sqrt(area / np.pi)
    tmp = np.zeros([ny, nx])
    tmp = mask_circle(tmp, radius=r, fillwith=1)
    tmp = frame_shift(tmp, ycom - (ny/2), xcom - (nx/2), imlib='opencv', border_mode='constant')  # no vip_fft because the image isn't square
    # measure translation
    shift_yx = phase_cross_correlation(tmp, shadow, upsample_factor=10)
    # express as a coordinate
    y = abs(shift_yx[0][0])
    x = abs(shift_yx[0][1])
    cy = np.round(ycom - y, 1)
    cx = np.round(xcom - x, 1)
    if verbose:
        print('The shadow has a {:.1f} px radius and is offset from the star in x,y by ({:.1f}, {:.1f}) px'.format(r, cx, cy), flush=True)
    if plot:
        plot_frames((median_frame, shadow, tmp), vmax=(np.percentile(median_frame, 99.5), 1, 1),
                    vmin=(np.percentile(median_frame, 0.5), 0, 0), label=('Median Lowpass Science', 'Inferred Shadow', 'Inferred Sky'),
                    dpi=300, circle=(int(xcom+cx), int(ycom+cy)), circle_radius=r,
                    label_color=('white', 'black', 'white'), cmap='inferno', top_colorbar=True,
                    horsp=0.2, save=self.outpath + 'shadow_fit.pdf')

    return r


def find_filtered_max(frame):
    """
    This method will find the location of the max after low pass filtering.
    It gives a rough approximation of the star's location, reliable in unsaturated frames where the star dominates.

    Parameters
    ----------
    frame : numpy array
        the frame in which to find the location of the star

    Returns
    ----------
    [ycom, xcom] : list
        location of AGPM or star
    """
    # apply low pass filter to help mind the brightest source
    frame = frame_filter_lowpass(frame, median_size=7, mode='median')
    frame = frame_filter_lowpass(frame, mode='gauss', fwhm_size=5)
    # obtain location of the bright source
    ycom, xcom = np.unravel_index(np.argmax(frame), frame.shape)
    return [ycom, xcom]

class raw_dataset:
    """
    In order to successfully run the pipeline you must run the methods in following order:
        1. dark_subtraction()
        2. flat_field_correction()
        3. correct_nan()
        4. correct_bad_pixels()
        5. first_frames_removal()
        6. get_stellar_psf()
        7. subtract_sky()
    This will prevent any missing files.
    """

    def __init__(self, inpath, outpath, dataset_dict, final_sz=None, coro=True):
        self.inpath = inpath
        self.outpath = outpath
        if not isdir(self.outpath):
            makedirs(self.outpath)
        self.final_sz = final_sz
        self.coro = coro
        sci_list = []
        # get the common size (crop size)
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])
        nx = open_fits(self.inpath + sci_list[0], verbose=False).shape[-1]
        self.com_sz = np.array([int(nx - 1)])
        write_fits(self.outpath + 'common_sz.fits', self.com_sz, verbose=False)
        # the size of the shadow in NACO data should be constant.
        # will differ for NACO data where the coronagraph has been adjusted
        self.shadow_r = 280  # shouldn't change for NaCO data
        sci_list_mjd = []  # observation time of each sci cube
        sky_list_mjd = []  # observation time of each sky cube
        with open(self.inpath + "sci_list_mjd.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list_mjd.append(float(line.split('\n')[0]))

        with open(self.inpath + "sky_list_mjd.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list_mjd.append(float(line.split('\n')[0]))
        self.sci_list_mjd = sci_list_mjd
        self.sky_list_mjd = sky_list_mjd
        self.dataset_dict = dataset_dict
        self.dit_unsat = dataset_dict['dit_unsat']
        self.fast_calibration = dataset_dict['fast_calibration']
        self.resel_ori = self.dataset_dict['wavelength'] * 206265 / (
                self.dataset_dict['size_telescope'] * self.dataset_dict['pixel_scale'])
        self.nproc = dataset_dict['nproc']

    def get_final_sz(self, final_sz=None, verbose=True, debug=False):
        """
        Update the cropping size as you wish.

        debug: enters Python debugger after finding the size
        """
        if final_sz is None:
            final_sz_ori = min(2 * self.crop_cen[0] - 1, 2 * self.crop_cen[1] - 1, 2 * \
                               (self.com_sz - self.crop_cen[0]) - 1, 2 * \
                               (self.com_sz - self.crop_cen[1]) - 1, int(2 * self.shadow_r))
        else:
            final_sz_ori = min(2 * self.crop_cen[0] - 1, 2 * self.crop_cen[1] - 1, \
                               2 * (self.com_sz - self.crop_cen[0]) - 1, \
                               2 * (self.com_sz - self.crop_cen[1]) - 1, \
                               int(2 * self.shadow_r), final_sz)
        if final_sz_ori % 2 == 0:
            final_sz_ori -= 1
        final_sz = int(final_sz_ori)  # iain: added int() around final_sz_ori as cropping requires an integer
        if verbose:
            print('The final crop size is {} px'.format(final_sz), flush=True)
        if debug:
            set_trace()
        return final_sz

    def dark_subtract(self, method='pca', npc_dark=1, imlib='vip-fft', NACO=True,
                      bad_quadrant=[3], verbose=True, debug=False, plot=True):
        """
        Dark subtraction of science, sky and flats using principal component analysis or median subtraction.
        Unsaturated frames are always median dark subtracted.
        All frames are also cropped to a common size.

        Most options are good as default, unless you have an additional bad detector quadrant.

        Parameters:
        ----------
        method : str, default = 'pca'
            'pca' for dark subtraction via principal component analysis
            'median' for classical median subtraction of dark
        npc_dark : int, optional
            Number of principal components subtracted during dark subtraction. Default = 1 (most variance in the PCA library)
        NACO : bool, optional
            If the NACO detector is being used or not. For masking known bad segments of the detector.
        bad_quadrant : list, optional
            List of bad quadrants to ignore. quadrants are in format  2 | 1  Default = 3 (inherently bad NaCO quadrant)
                                                                      3 | 4
        plot : bool, optional
            Whether to save plots to pdf.
        verbose : bool, optional
            Prints useful information.
        debug : bool, optional
            Prints significantly more information.
        """
        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

        if not isfile(self.inpath + sci_list[-1]):
            raise NameError('Missing OBJ files. Double check the contents of the input path')

        self.com_sz = int(open_fits(self.outpath + 'common_sz.fits', verbose=debug)[0])
        crop = 0
        if NACO:
            mask_std = np.zeros([self.com_sz, self.com_sz])
            cy, cx = frame_center(mask_std)
            # exclude the negative dot if the frame includes it
            if self.com_sz <= 733:
                mask_std[int(cy) - 23:int(cy) + 23, :] = 1
            else:
                crop = int((self.com_sz - 733) / 2)
                mask_std[int(cy) - 23:int(cy) + 23, :-crop] = 1
            write_fits(self.outpath + 'mask_std.fits', mask_std, verbose=debug)

        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        unsat_dark_list = []
        with open(self.inpath + "unsat_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_dark_list.append(line.split('\n')[0])

        flat_list = []
        with open(self.inpath + "flat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_list.append(line.split('\n')[0])

        flat_dark_list = []
        with open(self.inpath + "flat_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_dark_list.append(line.split('\n')[0])

        sci_dark_list = []
        with open(self.inpath + "sci_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_dark_list.append(line.split('\n')[0])

        pixel_scale = self.dataset_dict['pixel_scale']

        master_all_darks = []

        tmp = np.zeros([len(sci_dark_list), self.com_sz, self.com_sz])

        # cropping the SCI dark cubes to com_sz
        for sd, sd_name in enumerate(sci_dark_list):
            tmp_tmp = open_fits(self.inpath + sd_name, verbose=debug)
            n_dim = tmp_tmp.ndim
            if sd == 0:
                if n_dim == 2:
                    tmp_tmp = frame_crop(tmp_tmp, self.com_sz, force=True, 
                                         verbose=debug)
                    tmp = np.array([tmp_tmp])
                    master_all_darks.append(tmp_tmp)
                    if verbose:
                        print('Science dark dimensions: {}'.format(tmp_tmp.shape), flush=True)
                else:
                    tmp = cube_crop_frames(tmp_tmp, self.com_sz, force=True, verbose=debug)
                    for i in range(tmp.shape[0]):
                        master_all_darks.append(tmp[i])
                    if verbose:
                        print('Science dark dimensions: {}'.format(tmp[0].shape), flush=True)
            else:
                if n_dim == 2:
                    tmp = np.append(tmp, [frame_crop(tmp_tmp, self.com_sz, force=True, verbose=debug)], axis=0)
                    master_all_darks.append(tmp[-1])
                else:
                    tmp = np.append(tmp, cube_crop_frames(tmp_tmp, self.com_sz, force=True, verbose=debug), axis=0)
                    for i in range(tmp_tmp.shape[0]):
                        master_all_darks.append(tmp[-tmp_tmp.shape[0]+i])

        write_fits(self.outpath + 'sci_dark_cube.fits', tmp, verbose=debug)
        if verbose:
            print('Science dark cubes have been cropped and saved', flush=True)

        tmp = np.zeros([len(unsat_dark_list), self.com_sz, self.com_sz])

        # cropping of UNSAT dark frames to the common size or less
        # will only add to the master dark cube if it is the same size as the SKY and SCI darks
        for sd, sd_name in enumerate(unsat_dark_list):
            tmp_tmp = open_fits(self.inpath + sd_name, verbose=debug)
            n_dim = tmp_tmp.ndim
            if sd == 0:
                if n_dim == 2:
                    ny, nx = tmp_tmp.shape
                    if nx < self.com_sz:
                        tmp = np.array([frame_crop(tmp_tmp, nx - 1, force=True, verbose=debug)])
                    else:
                        if nx > self.com_sz:
                            tmp_tmp = frame_crop(tmp_tmp, self.com_sz, force=True, verbose=debug)
                        tmp = np.array([tmp_tmp])
                        master_all_darks.append(tmp_tmp)
                    if verbose:
                        print('Unsat dark dimensions: {}'.format(tmp_tmp.shape), flush=True)
                else:
                    nz, ny, nx = tmp_tmp.shape
                    if nx < self.com_sz:
                        tmp = cube_crop_frames(tmp_tmp, nx - 1, force=True, verbose=debug)
                    else:
                        if nx > self.com_sz:
                            tmp = cube_crop_frames(tmp_tmp, self.com_sz, force=True, verbose=debug)
                        else:
                            tmp = tmp_tmp
                        master_all_darks.append(np.median(tmp[-nz:], axis=0))
                    if verbose:
                        print('Unsat dark dimensions: {}'.format(tmp[-1].shape), flush=True)
            else:
                if n_dim == 2:
                    ny, nx = tmp_tmp.shape
                    if nx < self.com_sz:
                        tmp = np.append(tmp, [frame_crop(tmp_tmp, nx - 1, force=True, verbose=debug)], axis=0)
                    else:
                        if nx > self.com_sz:
                            tmp = np.append(tmp, [frame_crop(tmp_tmp, self.com_sz, force=True, verbose=debug)], axis=0)
                        else:
                            tmp = np.append(tmp, [tmp_tmp])
                        master_all_darks.append(tmp[-1])
                else:
                    nz, ny, nx = tmp_tmp.shape
                    if nx < self.com_sz:
                        tmp = np.append(tmp, cube_crop_frames(tmp_tmp, nx - 1, force=True, verbose=debug), axis=0)
                    else:
                        if nx > self.com_sz:
                            tmp = np.append(tmp, cube_crop_frames(tmp_tmp, self.com_sz, force=True, verbose=debug),
                                            axis=0)
                        else:
                            tmp = np.append(tmp, tmp_tmp)
                        master_all_darks.append(np.median(tmp[-nz:], axis=0))
        write_fits(self.outpath + 'unsat_dark_cube.fits', tmp, verbose=debug)
        if verbose:
            print('Unsat dark cubes have been cropped and saved', flush=True)

        # flat darks
        # cropping the flat dark cubes to com_sz
        tmp = np.zeros([len(flat_dark_list), self.com_sz, self.com_sz])
        for fd, fd_name in enumerate(flat_dark_list):
            tmp_tmp = open_fits(self.inpath + fd_name, verbose=debug)
            tmp[fd] = frame_crop(tmp_tmp, self.com_sz, force=True, verbose=debug)
            if (fd == 0) and verbose:
                print('Flat dark dimensions: {}'.format(tmp[fd].shape), flush=True)
            master_all_darks.append(tmp[fd])
        write_fits(self.outpath + 'flat_dark_cube.fits', tmp, verbose=debug)
        if verbose:
            print('Flat dark cubes from file have been cropped and saved', flush=True)

        if verbose:
            print('Total of {} median dark frames. Saving dark cube to fits file...'.format(len(master_all_darks)), flush=True)

        # convert master all darks to numpy array here
        master_all_darks = np.array(master_all_darks)
        write_fits(self.outpath + "master_all_darks.fits", master_all_darks, verbose=debug)

        # defining the mask for the sky/sci pca dark subtraction
        self.shadow_r = find_shadow_list(self, sci_list, threshold=None, plot=plot, verbose=verbose, debug=debug)

        if self.coro:
            self.crop_cen = find_AGPM(self.inpath + sci_list[0], verbose=verbose, debug=debug)
        else:
            if len(sky_list)>0:
                sub_list = sky_list
            else:
                sub_list = sci_dark_list
            tmp_tmp = []
            for sk, fits_name in enumerate(sub_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    tmp = np.median(tmp, axis=0)
                tmp_tmp.append(tmp)
            self.med_sky = np.median(np.array(tmp_tmp), axis=0)
            tmp_tmp = []
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    tmp = np.median(tmp, axis=0)
                tmp_tmp.append(tmp-self.med_sky)
            tmp_tmp = np.array(tmp_tmp)
            yx = approx_stellar_position(tmp_tmp, 
                                         int(self.resel_ori+1), 
                                         verbose=debug)
            self.crop_cen = (np.median(yx[:, 0]), np.median(yx[:, 1]))

        if plot:
            test_cube = open_fits(self.inpath + sci_list[0], verbose=debug)
            plot_frames(test_cube[-1], vmin=np.percentile(test_cube[-1], 0.5), vmax=np.percentile(test_cube[-1], 99.5),
                        cmap='inferno', dpi=300, circle_label='Inferred star/AGPM position', circle=(int(self.crop_cen[1]), int(self.crop_cen[0])),
                        circle_radius=3 * (self.resel_ori + 1), circle_alpha=1, label_size=8, label=sci_list[0],
                        save=self.outpath + 'Dark_subtract_inferred_star-AGPM_position.pdf')
            plt.close('all')

        mask_AGPM_com = np.ones([self.com_sz, self.com_sz])
        cy, cx = frame_center(mask_AGPM_com)

        inner_rad = 3 / pixel_scale
        outer_rad = self.shadow_r * 0.8

        if NACO:
            mask_sci = np.zeros([self.com_sz, self.com_sz])
            mask_sci[int(cy) - 23:int(cy) + 23, int(cx - outer_rad):int(cx + outer_rad)] = 1
            write_fits(self.outpath + 'mask_sci.fits', mask_sci, verbose=debug)

        # create mask for sci and sky
        mask_AGPM_com = get_annulus_segments(mask_AGPM_com, inner_rad, outer_rad - inner_rad, mode='mask')[0]
        mask_AGPM_com = frame_shift(mask_AGPM_com, self.crop_cen[0] - cy, self.crop_cen[1] - cx, border_mode='constant',
                                    imlib=imlib)
        # create mask for flats
        mask_AGPM_flat = np.ones([self.com_sz, self.com_sz])

        if verbose:
            print('The masks for SCI, SKY and FLAT have been defined', flush=True)
        # will exclude a quadrant if specified by looping over the list of bad quadrants and filling the mask with zeros
        if len(bad_quadrant) > 0:
            for quadrant in bad_quadrant:
                if quadrant == 1:
                    mask_AGPM_com[int(cy) + 1:, int(cx) + 1:] = 0
                    mask_AGPM_flat[int(cy) + 1:, int(cx) + 1:] = 0
                    # mask_std[int(cy)+1:,int(cx)+1:] = 0
                    # mask_sci[int(cy)+1:,int(cx)+1:] = 0
                if quadrant == 2:
                    mask_AGPM_com[int(cy) + 1:, :int(cx) + 1] = 0
                    mask_AGPM_flat[int(cy) + 1:, :int(cx) + 1] = 0
                    # mask_std[int(cy)+1:,:int(cx)+1] = 0
                    # mask_sci[int(cy)+1:,:int(cx)+1] = 0
                if quadrant == 3:
                    mask_AGPM_com[:int(cy) + 1, :int(cx) + 1] = 0
                    mask_AGPM_flat[:int(cy) + 1, :int(cx) + 1] = 0
                    # mask_std[:int(cy)+1,:int(cx)+1] = 0
                    # mask_sci[:int(cy)+1,:int(cx)+1] = 0
                if quadrant == 4:
                    mask_AGPM_com[:int(cy) + 1, int(cx) + 1:] = 0
                    mask_AGPM_flat[:int(cy) + 1, int(cx) + 1:] = 0
                    # mask_std[:int(cy)+1,int(cx)+1:] = 0
                    # mask_sci[:int(cy)+1,:int(cx)+1] = 0
        # save the mask for checking/testing
        write_fits(self.outpath + 'mask_AGPM_com.fits', mask_AGPM_com, verbose=debug)
        write_fits(self.outpath + 'mask_AGPM_flat.fits', mask_AGPM_flat, verbose=debug)
        write_fits(self.outpath + 'mask_std.fits', mask_std, verbose=debug)
        write_fits(self.outpath + 'mask_sci.fits', mask_sci, verbose=debug)
        if verbose:
            print('Masks have been saved as fits file', flush=True)

        def _plot_dark_median(fits_name, dark, before, after, file_type, outpath):
            if dark.ndim == 3:
                dark = np.median(dark, axis=0)
            if before.ndim == 3:
                before = np.median(before, axis=0)
            if after.ndim == 3:
                after = np.median(after, axis=0)
            plot_frames((dark, before, after), dpi=300,
                        vmax=(np.percentile(dark, 99.5), np.percentile(before, 99.5), np.percentile(after, 99.5)),
                        vmin=(np.percentile(dark, 0.5), np.percentile(before, 0.5), np.percentile(after, 0.5)),
                        label=('Median {} Dark'.format(file_type), 'Raw {} \n'.format(file_type) + fits_name, '{} Median Dark Subtracted'.format(file_type)),
                        cmap='inferno', top_colorbar=True, horsp=0.2,
                        save=outpath + '{}_median_dark_subtract.pdf'.format(file_type))
            plt.close('all')

        if method == 'median':
            # median dark subtraction of SCI cubes
            tmp_tmp_tmp = open_fits(self.outpath + 'sci_dark_cube.fits', verbose=debug)
            tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis=0)
            #tmp_tmp_tmp_median = np.median(tmp_tmp_tmp_median[np.where(mask_AGPM_com)])  # consider the median within the mask # this is a float!
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)
                    tmp_tmp = tmp.copy()
                    for i in range(tmp_tmp.shape[0]):
                        tmp_tmp[i] = tmp[i] - tmp_tmp_tmp_median
                else:
                    tmp = frame_crop(tmp, self.com_sz, force=True, verbose=debug)
                    tmp_tmp = tmp - tmp_tmp_tmp_median
                write_fits(self.outpath + '1_crop_' + fits_name, tmp_tmp, verbose=debug)
            if verbose:
                print('Dark has been median subtracted from SCI cubes', flush=True)

            if plot:
                _plot_dark_median(fits_name, tmp_tmp_tmp_median, tmp, tmp_tmp, 'Science', self.outpath)

            # median dark subtract of sky cubes
            tmp_tmp_tmp = open_fits(self.outpath + 'sci_dark_cube.fits', verbose=debug)
            tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis=0)
            #tmp_tmp_tmp_median = np.median(tmp_tmp_tmp_median[np.where(mask_AGPM_com)]) # this is a float!
            for sc, fits_name in enumerate(sky_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)
                    tmp_tmp = tmp.copy()
                    for i in range(tmp_tmp.shape[0]):
                        tmp_tmp[i] = tmp[i] - tmp_tmp_tmp_median
                else:
                    tmp = frame_crop(tmp, self.com_sz, force=True, verbose=debug)
                    tmp_tmp = tmp - tmp_tmp_tmp_median
                write_fits(self.outpath + '1_crop_' + fits_name, tmp_tmp, verbose=debug)
            if verbose:
                print('Dark has been median subtracted from SKY cubes', flush=True)
                
            if plot:
                _plot_dark_median(fits_name, tmp_tmp_tmp_median, tmp, tmp_tmp, 'Sky', self.outpath)

            # median dark subtract of flat cubes
            tmp_tmp = np.zeros([len(flat_list), self.com_sz, self.com_sz])
            tmp_tmp_tmp = open_fits(self.outpath + 'flat_dark_cube.fits', verbose=debug)
            tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis=0)
            #tmp_tmp_tmp_median = np.median(tmp_tmp_tmp_median[np.where(mask_AGPM_flat)]) # this is a float!
            for sc, fits_name in enumerate(flat_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 2:
                    tmp = frame_crop(tmp, self.com_sz, force=True, verbose=debug)
                else:
                    tmp = np.median(cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug), axis=0)
                tmp_tmp[sc] = tmp - tmp_tmp_tmp_median
            write_fits(self.outpath + '1_crop_flat_cube.fits', tmp_tmp, verbose=debug)
            if verbose:
                print('Dark has been median subtracted from FLAT frames', flush=True)

            if plot:
                _plot_dark_median(fits_name, tmp_tmp_tmp_median, tmp, tmp_tmp, 'Flat', self.outpath)

        # original code           ####################
        #        #now begin the dark subtraction using PCA
        #        npc_dark=1 #The ideal number of components to consider in PCA
        #
        #        #coordinate system for pca subtraction
        #        mesh = np.arange(0,self.com_sz,1)
        #        xv,yv = np.meshgrid(mesh,mesh)
        #
        #        tmp_tmp = np.zeros([len(flat_list),self.com_sz,self.com_sz])
        #        tmp_tmp_tmp = open_fits(self.outpath+'flat_dark_cube.fits')
        #        tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis = 0)
        #        #consider the difference in the medium of the frames without the lower left quadrant.
        #        tmp_tmp_tmp_median = tmp_tmp_tmp_median[np.where(np.logical_or(xv > cx, yv >  cy))]  # all but the bad quadrant in the bottom left
        #        diff = np.zeros([len(flat_list)])
        #        for fl, flat_name in enumerate(flat_list):
        #            tmp = open_fits(raw_path+flat_name, header=False, verbose=debug)
        #            #PCA works best if the flux is roughly on the same scale hence the difference is subtracted before PCA and added after.
        #            tmp_tmp[fl] = frame_crop(tmp, self.com_sz, force = True ,verbose=debug)
        #            tmp_tmp_tmp_tmp = tmp_tmp[fl]
        #            diff[fl] = np.median(tmp_tmp_tmp_median)-np.median(tmp_tmp_tmp_tmp[np.where(np.logical_or(xv > cx, yv >  cy))])
        #            tmp_tmp[fl]+=diff[fl]
        #        if debug:
        #            print('difference w.r.t dark = ',  diff)
        #        tmp_tmp_pca = cube_subtract_sky_pca(tmp_tmp, tmp_tmp_tmp,
        #                                    mask_AGPM_flat, ref_cube=None, ncomp=npc_dark)
        #        if debug:
        #            write_fits(self.outpath+'1_crop_flat_cube_diff.fits', tmp_tmp_pca)
        #        for fl, flat_name in enumerate(flat_list):
        #            tmp_tmp_pca[fl] = tmp_tmp_pca[fl]-diff[fl]
        #        write_fits(self.outpath+'1_crop_flat_cube.fits', tmp_tmp_pca)
        #        if verbose:
        #            print('Dark has been subtracted from FLAT cubes')
        # end original code       ###################

        # vals version of above
        #        npc_dark=1
        #        tmp_tmp = np.zeros([len(flat_list),self.com_sz,self.com_sz])
        #        tmp_tmp_tmp = open_fits(self.outpath+'flat_dark_cube.fits')
        #        npc_flat = tmp_tmp_tmp.shape[0] #not used?
        #        diff = np.zeros([len(flat_list)])
        #        for fl, flat_name in enumerate(flat_list):
        #            tmp = open_fits(raw_path+flat_name, header=False, verbose=False)
        #            tmp_tmp[fl] = frame_crop(tmp, self.com_sz, force = True, verbose=False)# added force = True
        #            write_fits(self.outpath+"TMP_flat_test_Val.fits",tmp_tmp[fl])
        #            #diff[fl] = np.median(tmp_tmp_tmp)-np.median(tmp_tmp[fl])
        #            #tmp_tmp[fl]+=diff[fl]
        #            tmp_tmp[fl] = tmp_tmp[fl] - bias
        #        print(diff)
        #        tmp_tmp_pca = cube_subtract_sky_pca(tmp_tmp, tmp_tmp_tmp - bias, mask_AGPM_flat, ref_cube=None, ncomp=npc_dark)
        #        for fl, flat_name in enumerate(flat_list):
        #            tmp_tmp_pca[fl] = tmp_tmp_pca[fl]-diff[fl]
        #        write_fits(self.outpath+'1_crop_flat_cube.fits', tmp_tmp_pca)
        #        if verbose:
        #            print('Dark has been subtracted from FLAT cubes')
        ###############

        ########### new Val code
        # create cube combining all darks
        #        master_all_darks = []
        #        #ntot_dark = len(sci_dark_list) + len(flat_dark_list) #+ len(unsat_dark_list)
        #        #master_all_darks = np.zeros([ntot_dark, self.com_sz, self.com_sz])
        #        tmp = open_fits(self.outpath + 'flat_dark_cube.fits', verbose = verbose)
        #
        #        # add each frame to the list
        #        for frame in tmp:
        #            master_all_darks.append(frame)
        #
        #        for idx,fname in enumerate(sci_dark_list):
        #            tmp = open_fits(self.inpath + fname, verbose=verbose)
        #            master_all_darks.append(tmp[-1])
        #
        #        #tmp = open_fits(self.outpath + 'sci_dark_cube.fits', verbose = verbose) # changed from master_sci_dark_cube.fits to sci_dark_cube.fits
        #
        #        #for frame in tmp:
        #        #    master_all_darks.append(frame)
        #
        #        if len(unsat_dark_list) > 0:
        #            for idx,fname in enumerate(unsat_dark_list):
        #                tmp = open_fits(self.inpath + fname, verbose=verbose)
        #                master_all_darks.append(tmp[-1])
        #            #tmp = open_fits(self.outpath + 'unsat_dark_cube.fits', verbose = verbose)
        #            #for frame in tmp:
        #                #master_all_darks.append(frame)
        #
        #        #master_all_darks[:len(flat_dark_list)] = tmp.copy()
        #        #master_all_darks[len(flat_dark_list):] = tmp.copy()
        if method == 'pca':
            def _plot_dark_pca(fits_name, dark, before, mask, after, file_type, outpath):
                if dark.ndim == 3:
                    dark = np.median(dark, axis=0)
                if before.ndim == 3:
                    before = np.median(before, axis=0)
                if after.ndim == 3:
                    after = np.median(after, axis=0)
                plot_frames((dark, before, mask, after), dpi=300,
                            vmax=(np.percentile(dark, 99.5), np.percentile(before, 99.5), 0, np.percentile(after, 99.5)),
                            vmin=(np.percentile(dark, 0.5), np.percentile(before, 0.5), 1, np.percentile(after, 0.5)),
                            label=('Median of all darks', 'Raw {} \n'.format(file_type) + fits_name, '{} Mask'.format(file_type),
                                   '{} PCA Dark Subtracted'.format(file_type)),
                            cmap='inferno', top_colorbar=True, horsp=0.2,
                            save=outpath + '{}_PCA_dark_subtract.pdf'.format(file_type))
                plt.close('all')

            # the cube of all darks - PCA works better with a larger library of darks
            tmp_tmp_tmp = open_fits(self.outpath + 'master_all_darks.fits', verbose=debug)
            tmp_tmp = np.zeros([len(flat_list), self.com_sz, self.com_sz])

            diff = np.zeros([len(flat_list)])
            bar = pyprind.ProgBar(len(flat_list), stream=1, title='Finding difference between DARKS and FLATS')
            for fl, flat_name in enumerate(flat_list):
                tmp = open_fits(self.inpath + flat_name, verbose=False)
                tmp_tmp[fl] = frame_crop(tmp, self.com_sz, force=True, verbose=False)  # added force = True
                diff[fl] = np.median(tmp_tmp_tmp) - np.median(
                    tmp_tmp[fl])  # median of pixels in all darks - median of all pixels in flat frame
                tmp_tmp[fl] += diff[fl]  # subtracting median of flat from the flat and adding the median of the dark
                bar.update()

            def _get_test_diff_flat(guess, verbose=False):
                # tmp_tmp_pca = np.zeros([self.com_sz,self.com_sz])
                # stddev = []
                # loop over values around the median of diff to scale the frames accurately
                # for idx,td in enumerate(test_diff):
                tmp_tmp_pca = np.median(cube_subtract_sky_pca(tmp_tmp + guess, tmp_tmp_tmp,
                                                              mask_AGPM_flat, ref_cube=None, ncomp=npc_dark), axis=0)
                tmp_tmp_pca -= np.median(
                    diff) + guess  # subtract the negative median of diff values and subtract test diff (aka add it back)
                subframe = tmp_tmp_pca[np.where(mask_std)]  # where mask_std is an optional argument
                # subframe = tmp_tmp_pca[int(cy)-23:int(cy)+23,:-17] # square around center that includes the bad lines in NaCO data
                # if idx ==0:
                subframe = subframe.reshape((-1, self.com_sz - crop))

                # stddev.append(np.std(subframe)) # save the stddev around this bad area
                stddev = np.std(subframe)
                write_fits(self.outpath + 'dark_flat_subframe.fits', subframe, verbose=debug)
                # if verbose:
                print('Guess = {}'.format(guess))
                print('Stddev = {}'.format(stddev), flush=True)

                #        for fl, flat_name in enumerate(flat_list):
                #            tmp_tmp_pca[fl] = tmp_tmp_pca[fl]-diff[fl]

                # return test_diff[np.argmin[stddev]] # value of test_diff corresponding to lowest stddev
                return stddev

            # step_size1 = 50
            # step_size2 = 10
            # n_test1 = 50
            # n_test2 = 50

            # lower_diff = guess - (n_test1 * step_size1) / 2
            # upper_diff = guess + (n_test1 * step_size1) / 2

            # test_diff = np.arange(lower_diff, upper_diff, n_test1) - guess
            # print('lower_diff:', lower_diff)
            # print('upper_diff:', upper_diff)
            # print('test_diff:', test_diff)
            # chisquare = function that computes stddev, p = test_diff
            # solu = minimize(chisquare, p, args=(cube, angs, etc.), method='Nelder-Mead', options=options)
            if verbose:
                print('FLATS difference w.r.t. DARKS:', diff)
                print('Calculating optimal PCA dark subtraction for FLATS...', flush=True)
            guess = 0
            solu = minimize(_get_test_diff_flat, x0=guess, args=(debug), method='Nelder-Mead', tol=2e-4,
                            options={'maxiter': 100, 'disp': verbose})

            # guess = solu.x
            # print('best diff:',guess)
            # # lower_diff = guess - (n_test2 * step_size2) / 2
            # # upper_diff = guess + (n_test2 * step_size2) / 2
            # #
            # # test_diff = np.arange(lower_diff, upper_diff, n_test2) - guess
            # # print('lower_diff:', lower_diff)
            # # print('upper_diff:', upper_diff)
            # # print('test_diff:', test_diff)
            #
            # solu = minimize(_get_test_diff_flat, x0=test_diff, args=(), method='Nelder-Mead',
            #                 options={'maxiter': 1})

            best_test_diff = solu.x  # x is the solution (ndarray)
            best_test_diff = best_test_diff[0]  # take out of array
            if verbose:
                print('Best difference (value) to add to FLATS is {} found in {} iterations'.format(best_test_diff,
                                                                                                    solu.nit), flush=True)

            # cond = True
            # max_it = 3 # maximum iterations
            # counter = 0
            # while cond and counter<max_it:
            #     index,best_diff = _get_test_diff_flat(self,first_guess = np.median(diff), n_test = n_test1,lower_limit = 0.1*np.median(diff),upper_limit = 2)
            #     if index !=0 and index !=n_test1-1:
            #         cond = False
            #     else:
            #         first_guess =
            #     counter +=1
            #     if counter==max_it:
            #         print('##### Reached maximum iterations for finding test diff! #####')
            # _,_ = _get_test_diff_flat(self, first_guess=best_diff, n_test=n_test2, lower_limit=0.8, upper_limit=1.2,plot=plot)

            # write_fits(self.outpath + '1_crop_flat_cube_test_diff.fits', tmp_tmp_pca + td, verbose=debug)
            # if verbose:
            #     print('stddev:', np.round(stddev, 3))
            #     print('Lowest standard dev is {} at frame {} with constant {}'.format(np.round(np.min(stddev), 2),
            #                                                                           np.round(np.argmin(stddev), 2) + 1,
            #                                                                           test_diff[np.argmin(stddev)]))

            tmp_tmp_pca = cube_subtract_sky_pca(tmp_tmp + best_test_diff, tmp_tmp_tmp,
                                                mask_AGPM_flat, ref_cube=None, ncomp=npc_dark)
            bar = pyprind.ProgBar(len(flat_list), stream=1, title='Correcting FLATS via PCA dark subtraction')
            for fl, flat_name in enumerate(flat_list):
                tmp_tmp_pca[fl] = tmp_tmp_pca[fl] - diff[fl] - best_test_diff  # add back the constant
                bar.update()
            write_fits(self.outpath + '1_crop_flat_cube.fits', tmp_tmp_pca, verbose=debug)

            if plot:
                _plot_dark_pca(flat_name, tmp_tmp_tmp, tmp_tmp, mask_AGPM_flat, tmp_tmp_pca, 'Flat', self.outpath)

            if verbose:
                print('Flats have been dark corrected', flush=True)

            # PCA dark subtraction of SCI cubes
            tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis=0)  # median frame of all darks
            tmp_tmp_tmp_median = np.median(
                tmp_tmp_tmp_median[np.where(mask_AGPM_com)])  # integer median of all the pixels within the mask

            tmp_tmp = np.zeros([len(sci_list), self.com_sz, self.com_sz])

            diff = np.zeros([len(sci_list)])
            bar = pyprind.ProgBar(len(sci_list), stream=1,
                                  title='Finding difference between DARKS and SCI cubes. This may take some time.')
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)  # open science
                tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)  # crop science to common size
                # PCA works best when the considering the difference
                tmp_median = np.median(tmp, axis=0)  # make median frame from all frames in cube
                # tmp_median = tmp_median[np.where(mask_AGPM_com)]
                diff[sc] = tmp_tmp_tmp_median - np.median(
                    tmp_median)  # median pixel value of all darks minus median pixel value of sci cube
                tmp_tmp[sc] = tmp_median + diff[sc]
                # if sc==0 or sc==middle_idx or sc==len(sci_list)-1:
                #     tmp_tmp[counter] = tmp_median + diff[sc]
                #     counter = counter + 1
                if debug:
                    print('difference w.r.t dark =', diff[sc], flush=True)
                bar.update()
            write_fits(self.outpath + 'dark_sci_diff.fits', diff, verbose=debug)
            write_fits(self.outpath + 'sci_plus_diff.fits', tmp_tmp, verbose=debug)
            # with open(self.outpath + "dark_sci_diff.txt", "w") as f:
            #     for diff_sci in diff:
            #         f.write(str(diff_sci) + '\n')
            if verbose:
                print('SCI difference w.r.t. DARKS has been saved to fits file.')
                print('SCI difference w.r.t. DARKS:', diff, flush=True)

            # lower_diff = 0.8*np.median(diff)
            # upper_diff = 1.2*np.median(diff)
            # test_diff = np.arange(abs(lower_diff),abs(upper_diff),50) - abs(np.median(diff)) # make a range of values in increments of 50 from 0.9 to 1.1 times the median
            # print('test diff:',test_diff)
            # tmp_tmp_pca = np.zeros([len(test_diff),self.com_sz,self.com_sz])
            # best_idx = []

            def _get_test_diff_sci(guess, verbose=False):
                # tmp_tmp_pca = np.zeros([self.com_sz,self.com_sz])
                # stddev = []
                # loop over values around the median of diff to scale the frames accurately
                # for idx,td in enumerate(test_diff):
                tmp_tmp_pca = np.median(cube_subtract_sky_pca(tmp_tmp + guess, tmp_tmp_tmp,
                                                              mask_AGPM_com, ref_cube=None, ncomp=npc_dark), axis=0)
                tmp_tmp_pca -= np.median(
                    diff) + guess  # subtract the negative median of diff values and subtract test diff (aka add it back)
                subframe = tmp_tmp_pca[np.where(mask_sci)]
                # subframe = tmp_tmp_pca[int(cy)-23:int(cy)+23,:-17] # square around center that includes the bad lines in NaCO data
                # if idx ==0:
                # stddev.append(np.std(subframe)) # save the stddev around this bad area
                stddev = np.std(subframe)
                if verbose:
                    print('Guess = {}'.format(guess))
                    print('Standard deviation = {}'.format(stddev), flush=True)
                subframe = subframe.reshape(46,
                                            -1)  # hard coded 46 because the subframe size is hardcoded to center pixel +-23
                write_fits(self.outpath + 'dark_sci_subframe.fits', subframe, verbose=debug)

                #        for fl, flat_name in enumerate(flat_list):
                #            tmp_tmp_pca[fl] = tmp_tmp_pca[fl]-diff[fl]

                # return test_diff[np.argmin[stddev]] # value of test_diff corresponding to lowest stddev
                return stddev

            # test_sci_list = [sci_list[i] for i in [0,middle_idx,-1]]

            # bar = pyprind.ProgBar(len(sci_list), stream=1, title='Testing diff for science cubes')
            guess = 0
            # best_diff = []
            # for sc in [0,middle_idx,-1]:
            if verbose:
                print('Calculating optimal PCA dark subtraction for SCI cubes. This may take some time.', flush=True)
            solu = minimize(_get_test_diff_sci, x0=guess, args=(verbose), method='Nelder-Mead', tol=2e-4,
                            options={'maxiter': 100, 'disp': verbose})

            best_test_diff = solu.x  # x is the solution (ndarray)
            best_test_diff = best_test_diff[0]  # take out of array
            # best_diff.append(best_test_diff)
            if verbose:
                print('Best difference (value) to add to SCI cubes is {} found in {} iterations'.format(best_test_diff,
                                                                                                        solu.nit), flush=True)
                # stddev = [] # to refresh the list after each loop
                # tmp = open_fits(self.inpath+sci_list[sc], header=False, verbose=debug)
                # tmp = cube_crop_frames(tmp, self.com_sz, force = True, verbose=debug)

                # for idx,td in enumerate(test_diff):
                # tmp_tmp_pca = np.median(cube_subtract_sky_pca(tmp_tmp[sc]+guess, tmp_tmp_tmp,mask_AGPM_com, ref_cube=None, ncomp=npc_dark),axis=0)
                # tmp_tmp_pca-= np.median(diff)+td
                # subframe = tmp_tmp_pca[np.where(mask_std)]
                # subframe = tmp_tmp_pca[idx,int(cy)-23:int(cy)+23,:] # square around center that includes that bad lines
                # stddev.append(np.std(subframe))
                # best_idx.append(np.argmin(stddev))
                # print('Best index of test diff: {} of constant: {}'.format(np.argmin(stddev),test_diff[np.argmin(stddev)]))
                # bar.update()
                # if sc == 0:
                #    write_fits(self.outpath+'1_crop_sci_cube_test_diff.fits', tmp_tmp_pca + td, verbose = debug)

            # sci_list_mjd = np.array(self.sci_list_mjd) # convert list to numpy array
            # xp = sci_list_mjd[np.array([0,middle_idx,-1])] # only get first, middle, last
            # #fp = test_diff[np.array(best_idx)]
            # fp = best_diff
            # opt_diff = np.interp(x = sci_list_mjd, xp = xp, fp = fp, left=None, right=None, period=None) # optimal diff for each sci cube

            if verbose:
                print('Optimal constant to apply to each science cube: {}'.format(best_test_diff), flush=True)

            bar = pyprind.ProgBar(len(sci_list), stream=1, title='Correcting SCI cubes via PCA dark subtraction')
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)

                tmp_tmp_pca = cube_subtract_sky_pca(tmp + diff[sc] + best_test_diff, tmp_tmp_tmp,
                                                    mask_AGPM_com, ref_cube=None, ncomp=npc_dark)

                tmp_tmp_pca = tmp_tmp_pca - diff[sc] - best_test_diff  # add back the constant
                write_fits(self.outpath + '1_crop_' + fits_name, tmp_tmp_pca, verbose=debug)
                bar.update()

            if plot:
                _plot_dark_pca(fits_name, tmp_tmp_tmp, tmp, mask_AGPM_com, tmp_tmp_pca, 'Science', self.outpath)

            if verbose:
                print('Dark has been subtracted from SCI cubes', flush=True)

            # dark subtract of sky cubes
            # tmp_tmp_tmp = open_fits(self.outpath+'sci_dark_cube.fits')
            #        tmp_tmp_tmp = open_fits(self.outpath+'master_all_darks.fits')
            #        tmp_tmp_tmp_median = np.median(tmp_tmp_tmp,axis = 0)
            #        tmp_tmp_tmp_median = np.median(tmp_tmp_tmp_median[np.where(mask_AGPM_com)])
            #
            #        bar = pyprind.ProgBar(len(sky_list), stream=1, title='Correcting dark current in sky cubes')
            #        for sc, fits_name in enumerate(sky_list):
            #            tmp = open_fits(self.inpath+fits_name, header=False, verbose=debug)
            #            tmp = cube_crop_frames(tmp, self.com_sz, force = True, verbose=debug)
            #            tmp_median = np.median(tmp,axis = 0)
            #            tmp_median = tmp_median[np.where(mask_AGPM_com)]
            #            diff = tmp_tmp_tmp_median - np.median(tmp_median)
            #            if debug:
            #                   print('difference w.r.t dark = ',  diff)
            #            tmp_tmp = cube_subtract_sky_pca(tmp +diff +test_diff[np.argmin(stddev)], tmp_tmp_tmp,
            #                                    mask_AGPM_com, ref_cube=None, ncomp=npc_dark)
            #            if debug:
            #                write_fits(self.outpath+'1_crop_diff'+fits_name, tmp_tmp)
            #            write_fits(self.outpath+'1_crop_'+fits_name, tmp_tmp -diff -test_diff[np.argmin(stddev)], verbose = debug)
            #            bar.update()
            #        if verbose:
            #            print('Dark has been subtracted from SKY cubes')
            #        if plot:
            #            tmp = np.median(tmp, axis = 0)
            #            tmp_tmp = np.median(tmp_tmp-diff,axis = 0)
            #        if plot == 'show':
            #            plot_frames((tmp,tmp_tmp,mask_AGPM_com), vmax = (25000,25000,1), vmin = (-2500,-2500,0))
            #        if plot == 'save':
            #            plot_frames((tmp,tmp_tmp,mask_AGPM_com), vmax = (25000,25000,1), vmin = (-2500,-2500,0),save = self.outpath + 'SKY_PCA_dark_subtract')

            tmp_tmp_tmp = open_fits(self.outpath + 'master_all_darks.fits', verbose=debug)
            tmp_tmp_tmp_median = np.median(tmp_tmp_tmp, axis=0)  # median frame of all darks
            tmp_tmp_tmp_median = np.median(
                tmp_tmp_tmp_median[np.where(mask_AGPM_com)])  # integer median of all the pixels within the mask

            tmp_tmp = np.zeros([len(sky_list), self.com_sz, self.com_sz])

            diff = np.zeros([len(sky_list)])

            bar = pyprind.ProgBar(len(sky_list), stream=1, title='Finding difference between darks and sky cubes')
            for sc, fits_name in enumerate(sky_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)  # open sky
                tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)  # crop sky to common size
                # PCA works best when the considering the difference
                tmp_median = np.median(tmp, axis=0)  # make median frame from all frames in cube
                # tmp_median = tmp_median[np.where(mask_AGPM_com)]
                diff[sc] = tmp_tmp_tmp_median - np.median(
                    tmp_median)  # median pixel value of all darks minus median pixel value of sky cube
                tmp_tmp[sc] = tmp_median + diff[sc]
                if debug:
                    print('difference w.r.t dark =', diff[sc], flush=True)
                bar.update()
            write_fits(self.outpath + 'dark_sci_diff.fits', diff, verbose=debug)
            if verbose:
                print('SKY difference w.r.t. DARKS has been saved to fits file.')
                print('SKY difference w.r.t. DARKS:', diff, flush=True)

            def _get_test_diff_sky(guess, verbose=False):
                # tmp_tmp_pca = np.zeros([self.com_sz,self.com_sz])
                # stddev = []
                # loop over values around the median of diff to scale the frames accurately
                # for idx,td in enumerate(test_diff):
                tmp_tmp_pca = np.median(cube_subtract_sky_pca(tmp_tmp + guess, tmp_tmp_tmp,
                                                              mask_AGPM_com, ref_cube=None, ncomp=npc_dark), axis=0)
                tmp_tmp_pca -= np.median(
                    diff) + guess  # subtract the negative median of diff values and subtract test diff (aka add it back)
                subframe = tmp_tmp_pca[np.where(mask_sci)]
                # subframe = tmp_tmp_pca[int(cy)-23:int(cy)+23,:-17] # square around center that includes the bad lines in NaCO data
                # if idx ==0:
                # stddev.append(np.std(subframe)) # save the stddev around this bad area
                stddev = np.std(subframe)
                if verbose:
                    print('Guess = {}'.format(guess))
                    print('Standard deviation = {}'.format(stddev), flush=True)
                subframe = subframe.reshape(46,
                                            -1)  # hard coded 46 because the subframe size is hardcoded to center pixel +-23
                write_fits(self.outpath + 'dark_sky_subframe.fits', subframe, verbose=debug)

                #        for fl, flat_name in enumerate(flat_list):
                #            tmp_tmp_pca[fl] = tmp_tmp_pca[fl]-diff[fl]

                # return test_diff[np.argmin[stddev]] # value of test_diff corresponding to lowest stddev
                return stddev

            guess = 0
            if verbose:
                print('Calculating optimal PCA dark subtraction for SKY cubes. This may take some time.', flush=True)
            solu = minimize(_get_test_diff_sky, x0=guess, args=(verbose), method='Nelder-Mead', tol=2e-4,
                            options={'maxiter': 100, 'disp': verbose})

            best_test_diff = solu.x  # x is the solution (ndarray)
            best_test_diff = best_test_diff[0]  # take out of array

            #
            # lower_diff = 0.9*np.median(diff)
            # upper_diff = 1.1*np.median(diff)
            # test_diff = np.arange(abs(lower_diff),abs(upper_diff),50) - abs(np.median(diff)) # make a range of values in increments of 50 from 0.9 to 1.1 times the median
            # tmp_tmp_pca = np.zeros([len(test_diff),self.com_sz,self.com_sz])
            # best_idx = []

            # middle_idx = int(len(sky_list)/2)

            # print('Testing diff for SKY cubes')
            # for sc in [0,middle_idx,-1]:
            #     stddev = [] # to refresh the list after each loop
            #     tmp = open_fits(self.inpath+sky_list[sc], header=False, verbose=debug)
            #     tmp = cube_crop_frames(tmp, self.com_sz, force = True, verbose=debug)
            #
            #     for idx,td in enumerate(test_diff):
            #         tmp_tmp_pca[idx] = np.median(cube_subtract_sky_pca(tmp+diff[sc]+td, tmp_tmp_tmp,
            #                                                 mask_AGPM_com, ref_cube=None, ncomp=npc_dark),axis=0)
            #         tmp_tmp_pca[idx]-= np.median(diff)+td
            #
            #         subframe = tmp_tmp_pca[idx,int(cy)-23:int(cy)+23,:] # square around center that includes that bad lines
            #         stddev.append(np.std(subframe))
            #     best_idx.append(np.argmin(stddev))
            #     print('Best index of test diff: {} of constant: {}'.format(np.argmin(stddev),test_diff[np.argmin(stddev)]))
            #     #bar.update()
            #     if sc == 0:
            #         write_fits(self.outpath+'1_crop_sky_cube_test_diff.fits', tmp_tmp_pca + td, verbose = debug)
            # print('test')
            # sky_list_mjd = np.array(self.sky_list_mjd) # convert list to numpy array
            # xp = sky_list_mjd[np.array([0,middle_idx,-1])] # only get first, middle, last
            # fp = test_diff[np.array(best_idx)]
            #
            # opt_diff = np.interp(x = sky_list_mjd, xp = xp, fp = fp, left=None, right=None, period=None) # optimal diff for each sci cube
            # print('Opt diff',opt_diff)
            # if debug:
            #     with open(self.outpath+"best_idx_sky.txt", "w") as f:
            #         for idx in best_idx:
            #             f.write(str(idx)+'\n')
            # if verbose:
            #     print('Optimal constant: {}'.format(opt_diff))
            if verbose:
                print('Optimal constant to apply to each sky cube: {}'.format(best_test_diff), flush=True)

            bar = pyprind.ProgBar(len(sky_list), stream=1, title='Correcting SKY cubes via PCA dark subtraction')
            for sc, fits_name in enumerate(sky_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                tmp = cube_crop_frames(tmp, self.com_sz, force=True, verbose=debug)

                tmp_tmp_pca = cube_subtract_sky_pca(tmp + diff[sc] + best_test_diff, tmp_tmp_tmp,
                                                    mask_AGPM_com, ref_cube=None, ncomp=npc_dark)

                tmp_tmp_pca = tmp_tmp_pca - diff[sc] - best_test_diff  # add back the constant
                write_fits(self.outpath + '1_crop_' + fits_name, tmp_tmp_pca, verbose=debug)
                bar.update()

            if plot:
                _plot_dark_pca(fits_name, tmp_tmp_tmp, tmp, mask_AGPM_com, tmp_tmp_pca, 'Sky', self.outpath)

        # median dark subtract of UNSAT cubes
        if len(unsat_list)>0:
            tmp_tmp_tmp = open_fits(self.outpath + 'unsat_dark_cube.fits', verbose=debug)
            tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis=0)
            # no need to crop the unsat frame at the same size as the sci images if they are smaller
            bar = pyprind.ProgBar(len(unsat_list), stream=1, title='Correcting dark current in unsaturated cubes')
            for un, fits_name in enumerate(unsat_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.shape[2] > self.com_sz:
                    nx_unsat_crop = self.com_sz
                    tmp = cube_crop_frames(tmp, nx_unsat_crop, force=True, verbose=debug)
                    tmp_tmp = tmp - tmp_tmp_tmp
                elif tmp.shape[2] % 2 == 0:
                    nx_unsat_crop = tmp.shape[2] - 1
                    tmp = cube_crop_frames(tmp, nx_unsat_crop, force=True, verbose=debug)
                    tmp_tmp = tmp - tmp_tmp_tmp
                else:
                    nx_unsat_crop = tmp.shape[2]
                    tmp_tmp = tmp - tmp_tmp_tmp
                write_fits(self.outpath + '1_crop_unsat_' + fits_name, tmp_tmp, verbose=debug)
                bar.update()
    
            if verbose:
                print('Dark has been subtracted from UNSAT cubes', flush=True)

            if plot:
                _plot_dark_median(fits_name, tmp_tmp_tmp, tmp, tmp_tmp, 'Unsat', self.outpath)

    def flat_field_correction(self, verbose=True, plot=True, debug=False, remove=False):
        """
        Scaling of the cubes according to the FLATS, in order to minimise any bias in the pixels.
        Can handle the case when there is no airmass in the FITS header.

        verbose: bool, optional
            Prints completion messages.
        plot: bool, optional
            Save before/after plots.
        debug: bool, optional
            Prints significantly more information.
        remove: bool, optional
            Cleans files for unused FITS.
        """
        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

        if not isfile(self.outpath + '1_crop_' + sci_list[-1]):
            raise NameError('Missing 1_crop_*.fits. Run: dark_subtract()')

        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

        flat_list = []
        with open(self.inpath + "flat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_list.append(line.split('\n')[0])

        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        self.com_sz = int(open_fits(self.outpath + 'common_sz', verbose=debug)[0])

        flat_airmass_test = []
        tmp, header = open_fits(self.inpath + flat_list[0], header=True, verbose=debug)
        # attempt to get the airmass from the header
        try:
            flat_airmass_test.append(header['AIRMASS'])
        except:
            print('###### No AIRMASS detected in header!!! Inferring airmass .... ######', flush=True)

        flat_X = []
        flat_X_values = []

        if len(flat_airmass_test) > 0 or len(flat_list) == 15:
            if len(flat_airmass_test) > 0:  # if the airmass exists, we can group the flats based on airmass
                if verbose:
                    print('AIRMASS detected in FLATS header. Grouping FLATS by airmass ....', flush=True)
                # flat cubes measured at 3 different airmass
                for fl, flat_name in enumerate(flat_list):
                    tmp, header = open_fits(self.inpath + flat_list[fl], header=True, verbose=debug)
                    flat_X.append(header['AIRMASS'])
                    if fl == 0:
                        flat_X_values.append(header['AIRMASS'])
                    else:
                        list_occ = [np.isclose(header['AIRMASS'], x, atol=0.1) for x in
                                    flat_X_values]  # sorts nearby values together
                        if True not in list_occ:
                            flat_X_values.append(header['AIRMASS'])
                flat_X_values = np.sort(flat_X_values)  # !!! VERY IMPORTANT, DO NOT COMMENT
                if verbose:
                    print('Airmass values in FLATS: {}'.format(flat_X_values))
                    print('The airmass values have been sorted into a list', flush=True)

            # if no airmass in header, we can group by using the median pixel value across the flat
            elif len(flat_list) == 15:
                # use same structure as above, replacing airmass with median background level
                for fl, flat_name in enumerate(flat_list):
                    tmp = open_fits(self.inpath + flat_list[fl], verbose=debug)
                    flat_X.append(np.median(tmp))
                    if fl == 0:
                        flat_X_values.append(np.median(tmp))
                    else:
                        list_occ = [np.isclose(np.median(tmp), x, atol=50) for x in flat_X_values]
                        if True not in list_occ:
                            flat_X_values.append(np.median(tmp))
                flat_X_values = np.sort(flat_X_values)
                if verbose:
                    print('Median FLAT values: {}'.format(flat_X_values))
                    print('The median FLAT values have been sorted into a list', flush=True)

            # There should be 15 twilight flats in total with NACO; 5 at each airmass. BUG SOMETIMES!
            flat_tmp_cube_1 = np.zeros([5, self.com_sz, self.com_sz])
            flat_tmp_cube_2 = np.zeros([5, self.com_sz, self.com_sz])
            flat_tmp_cube_3 = np.zeros([5, self.com_sz, self.com_sz])
            counter_1 = 0
            counter_2 = 0
            counter_3 = 0
    
            flat_cube_nX = np.zeros([3, self.com_sz, self.com_sz])
    
            # TAKE MEDIAN OF each group of 5 frames with SAME AIRMASS
            flat_cube = open_fits(self.outpath + '1_crop_flat_cube.fits', verbose=debug)
            for fl, self.flat_name in enumerate(flat_list):
                if find_nearest(flat_X_values, flat_X[fl]) == 0:
                    flat_tmp_cube_1[counter_1] = flat_cube[fl]
                    counter_1 += 1
                elif find_nearest(flat_X_values, flat_X[fl]) == 1:
                    flat_tmp_cube_2[counter_2] = flat_cube[fl]
                    counter_2 += 1
                elif find_nearest(flat_X_values, flat_X[fl]) == 2:
                    flat_tmp_cube_3[counter_3] = flat_cube[fl]
                    counter_3 += 1
    
            flat_cube_nX[0] = np.median(flat_tmp_cube_1, axis=0)
            flat_cube_nX[1] = np.median(flat_tmp_cube_2, axis=0)
            flat_cube_nX[2] = np.median(flat_tmp_cube_3, axis=0)
            if verbose:
                print('The median FLAT cubes with same airmass have been defined', flush=True)
                
        else:  # if not 15 frames, manually sort by median pixel value
            msg = '{} (!=15) flat-fields found => assuming old way of calculating flats'
            print(msg.format(len(flat_list)), flush=True)
            all_flats = []
            flat_cube = open_fits(self.outpath + '1_crop_flat_cube.fits', verbose=debug)
            for fl, flat_name in enumerate(flat_list):
                tmp = open_fits(self.inpath + flat_list[fl], verbose=debug)
                flat_X.append(np.median(tmp))
                all_flats.append(flat_cube[fl])
            sort_idx = np.argsort(flat_X)
            all_flats = np.array(all_flats)
            flat_cube_nX = all_flats[sort_idx]

        # add linear regression to determine dark for each pixel

        # create master flat field
        n_fl = flat_cube_nX.shape[0]
        med_fl = np.zeros(n_fl)
        gains_all = np.zeros([n_fl, self.com_sz, self.com_sz])
        for ii in range(n_fl):
            med_fl[ii] = np.nanmedian(flat_cube_nX[ii])
            gains_all[ii] = flat_cube_nX[ii] / med_fl[ii]
        master_flat_frame = np.nanmedian(gains_all, axis=0)
        write_fits(self.outpath + 'master_flat_field.fits', master_flat_frame, verbose=debug)
        ## create unsat master flat field
        if len(unsat_list) > 0:
            tmp = open_fits(self.outpath + '1_crop_unsat_' + unsat_list[-1], verbose=debug)
            nx_unsat_crop = tmp.shape[2]
            if nx_unsat_crop < master_flat_frame.shape[1]:
                master_flat_unsat = frame_crop(master_flat_frame, nx_unsat_crop, verbose=debug)
            else:
                master_flat_unsat = master_flat_frame
            write_fits(self.outpath + 'master_flat_field_unsat.fits', master_flat_unsat, verbose=debug)
        if verbose:
            print('Master flat frames has been saved', flush=True)
        if plot:
            if len(unsat_list) > 0:
                plot_frames((master_flat_frame, master_flat_unsat),
                            vmax=(np.percentile(master_flat_frame, 99.5), np.percentile(master_flat_unsat, 99.5)),
                            vmin=(np.percentile(master_flat_frame, 0.5), np.percentile(master_flat_unsat, 0.5)),
                            dpi=300, label=('Master flat frame', 'Master flat unsat'), cmap='inferno', top_colorbar=True,
                            horsp=0.2, save=self.outpath + 'Master_flat.pdf')
            else:
                plot_frames(master_flat_frame, vmax=np.percentile(master_flat_frame, 99.5),
                            vmin=np.percentile(master_flat_frame, 0.5), dpi=300, label='Master flat frame',
                            cmap='inferno', save=self.outpath + 'Master_flat.pdf')
            plt.close('all')

        def _plot_flat(fits_name, flat, before, after, file_type, outpath):
            if flat.ndim == 3:
                flat = np.nanmedian(flat, axis=0)
            if before.ndim == 3:
                before = np.nanmedian(before, axis=0)
            if after.ndim == 3:
                after = np.nanmedian(after, axis=0)
            plot_frames((flat, before, after), dpi=300,
                        vmax=(np.percentile(flat, 99.5), np.percentile(before, 99.5), np.percentile(after, 99.5)),
                        vmin=(np.percentile(flat, 0.5), np.percentile(before, 0.5), np.percentile(after, 0.5)),
                        label=('Master Flat', 'Raw {} \n'.format(file_type) + fits_name, '{} Flat Fielded \n'.format(file_type) + fits_name),
                        cmap='inferno', top_colorbar=True, horsp=0.2, save=outpath + '{}_flat_corrected.pdf'.format(file_type))
            plt.close('all')

        # scaling of SCI cubes with respect to the master flat
        bar = pyprind.ProgBar(len(sci_list), stream=1, title='Scaling SCI cubes with respect to the master flat')
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '1_crop_' + fits_name, verbose=debug)
            if tmp.ndim==3:
                tmp_tmp = np.zeros_like(tmp)
                for jj in range(tmp.shape[0]):
                    tmp_tmp[jj] = tmp[jj] / master_flat_frame
            else:
                tmp_tmp = tmp / master_flat_frame
            write_fits(self.outpath + '2_ff_' + fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if remove:
                system("rm " + self.outpath + '1_crop_' + fits_name)
        if verbose:
            print('Done scaling SCI frames with respect to FLAT', flush=True)
        if plot:
            _plot_flat(fits_name, master_flat_frame, tmp, tmp_tmp, 'Science', self.outpath)

        # scaling of SKY cubes with respect to the master flat
        bar = pyprind.ProgBar(len(sky_list), stream=1, title='Scaling SKY cubes with respect to the master flat')
        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath + '1_crop_' + fits_name, verbose=debug)
            if tmp.ndim==3:
                tmp_tmp = np.zeros_like(tmp)
                for jj in range(tmp.shape[0]):
                    tmp_tmp[jj] = tmp[jj] / master_flat_frame
            else:
                tmp_tmp = tmp / master_flat_frame
            write_fits(self.outpath + '2_ff_' + fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if remove:
                system("rm " + self.outpath + '1_crop_' + fits_name)
        if verbose:
            print('Done scaling SKY frames with respect to FLAT', flush=True)
        if plot:
            _plot_flat(fits_name, master_flat_frame, tmp, tmp_tmp, 'Sky', self.outpath)

        # scaling of UNSAT cubes with respect to the master flat unsat
        if len(unsat_list) > 0:
            bar = pyprind.ProgBar(len(unsat_list), stream=1, title='Scaling UNSAT cubes with respect to the master flat')
            for un, fits_name in enumerate(unsat_list):
                tmp = open_fits(self.outpath + '1_crop_unsat_' + fits_name, verbose=debug)
                tmp_tmp = np.zeros_like(tmp)
                for jj in range(tmp.shape[0]):
                    tmp_tmp[jj] = tmp[jj] / master_flat_unsat
                write_fits(self.outpath + '2_ff_unsat_' + fits_name, tmp_tmp, verbose=debug)
                bar.update()
                if remove:
                    system("rm " + self.outpath + '1_crop_unsat_' + fits_name)
            if verbose:
                print('Done scaling UNSAT frames with respect to FLAT', flush=True)
            if plot:
                _plot_flat(fits_name, master_flat_unsat, tmp, tmp_tmp, 'Unsat', self.outpath)

    def correct_nan(self, verbose=True, debug=False, remove=False):
        """
        Corrects NAN pixels in cubes, if present.

        verbose: bool, optional
            Prints completion messages.
        debug: bool, optional
            Prints significantly more information.
        remove: bool, optional
            Cleans files for unused FITS.
        """
        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

        if not isfile(self.outpath + '2_ff_' + sci_list[-1]):
            raise NameError('Missing 2_ff_*.fits. Run: flat_field_correction()')

        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        n_sci = len(sci_list)
        n_sky = len(sky_list)
        n_unsat = len(unsat_list)

        bar = pyprind.ProgBar(n_sci, stream=1, title='Correcting NaN pixels in SCI frames')
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '2_ff_' + fits_name, verbose=debug)
            tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug, nproc=self.nproc)
            write_fits(self.outpath + '2_nan_corr_' + fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if remove:
                system("rm " + self.outpath + '2_ff_' + fits_name)
        if verbose:
            print('Done correcting NaN pixels in SCI frames', flush=True)

        bar = pyprind.ProgBar(n_sky, stream=1, title='Correcting NaN pixels in SKY frames')
        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath + '2_ff_' + fits_name, verbose=debug)
            tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug, nproc=self.nproc)
            write_fits(self.outpath + '2_nan_corr_' + fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if remove:
                system("rm " + self.outpath + '2_ff_' + fits_name)
        if verbose:
            print('Done correcting NaN pixels in SKY frames', flush=True)

        if len(unsat_list) > 0:
            bar = pyprind.ProgBar(n_unsat, stream=1, title='Correcting NaN pixels in UNSAT frames')
            for un, fits_name in enumerate(unsat_list):
                tmp = open_fits(self.outpath + '2_ff_unsat_' + fits_name, verbose=debug)
                tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug, nproc=self.nproc)
                write_fits(self.outpath + '2_nan_corr_unsat_' + fits_name, tmp_tmp, verbose=debug)
                bar.update()
                if remove:
                    system("rm " + self.outpath + '2_ff_unsat_' + fits_name)
            if verbose:
                print('Done correcting NaN pixels in UNSAT frames', flush=True)

    def correct_bad_pixels(self, verbose=True,  plot=True, overwrite=False, debug=False, remove=False):
        """
        Correct bad pixels twice. First correction is for the bad pixels determined from the flat fields.
        Another correction is needed to correct bad pixels in each frame caused by residuals, hot pixels and gamma-rays.

        plot: bool, optional
            Save relevant plots.
        remove: bool, optional
            Cleans files for unused FITS.
        """

        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

        if not isfile(self.outpath + '2_nan_corr_' + sci_list[-1]):
            raise NameError('Missing 2_nan_corr_*.fits. Run: correct_nan_pixels()')

        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        self.com_sz = int(open_fits(self.outpath + 'common_sz.fits', verbose=debug)[0])

        master_flat_frame = open_fits(self.outpath + 'master_flat_field.fits', verbose=debug)
        # Create bpix map
        bpix = np.where(np.abs(master_flat_frame - 1.09) > 0.41)  # i.e. for QE < 0.68 and QE > 1.5
        bpix_map = np.zeros([self.com_sz, self.com_sz])
        bpix_map[bpix] = 1
        write_fits(self.outpath + 'master_bpix_map.fits', bpix_map, verbose=debug)

        if len(unsat_list) > 0:
            tmp = open_fits(self.outpath + '2_nan_corr_unsat_' + unsat_list[-1], verbose=debug)
            nx_unsat_crop = tmp.shape[2]        
            if nx_unsat_crop < bpix_map.shape[1]:
                bpix_map_unsat = frame_crop(bpix_map, nx_unsat_crop, force=True, verbose=debug)
            else:
                bpix_map_unsat = bpix_map
            write_fits(self.outpath + 'master_bpix_map_unsat.fits', bpix_map_unsat, verbose=debug)
            if plot:
                plot_frames((bpix_map, bpix_map_unsat), vmin=(0, 0), vmax=(1, 1), cmap='inferno', top_colorbar=True,
                            horsp=0.2, label=('Master Science/Sky Bad Pixel Map \nIdentified from flats', 'Master Unsat Bad Pixel Map \nIdentified from flats'),
                            save=self.outpath + 'Master_bpix_map.pdf')
            
        elif plot:
            plot_frames(bpix_map, vmin=0, vmax=1, cmap='inferno', horsp=0.2, label='Master Science/Sky Bad Pixel Map \nIdentified from flats',
                        save=self.outpath + 'Master_bpix_map.pdf')

        if verbose:
            # number of bad pixels
            nbpix = int(np.sum(bpix_map))
            ntotpix = self.com_sz ** 2
            print("Total number of bpix: ", nbpix)
            print("Total number of pixels: ", ntotpix)
            print("=> {:.5f}% are bad pixels.".format(100 * nbpix / ntotpix), flush=True)

        # update final crop size
        if self.coro:
            self.crop_cen = find_AGPM(self.outpath + '2_nan_corr_' + sci_list[0], verbose=verbose, debug=debug)
            self.final_sz = self.get_final_sz(self.final_sz, verbose=verbose, debug=debug)
        else:
            tmp_tmp = []
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.inpath + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    tmp = np.median(tmp, axis=0)
                tmp_tmp.append(tmp-self.med_sky)
            tmp_tmp = np.array(tmp_tmp)
            yx = approx_stellar_position(tmp_tmp, int(self.resel_ori+1), verbose=debug)
            min_final_sz = int(2*np.amax([np.amax(yx[:,0])-np.amin(yx[:,0]),
                                          np.amax(yx[:,1])-np.amax(yx[:,1])])+1)
            min_final_sz = min(min_final_sz, self.com_sz-2)
            max_final_sz = int(2*min(np.amin(yx), self.com_sz-np.amax(yx)))
                                     
            if self.final_sz is not None:
                # consider max because of potential dithering
                self.final_sz = min(max(self.final_sz, min_final_sz), 
                                    max_final_sz)
            else:
                self.final_sz = min(self.com_sz-2, max_final_sz)
            if self.final_sz%2 != self.com_sz%2:
                self.final_sz -= 1
            self.crop_cen = [np.nanmean(yx[:,0]), np.nanmean(yx[:,1])]
            
        self.crop_cen = [self.crop_cen[1], self.crop_cen[0]]
        write_fits(self.outpath + 'final_sz.fits', np.array([self.final_sz]), verbose=debug)

        # crop frames to that size
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '2_nan_corr_' + fits_name, verbose=debug)
            if sc == 0:
                if tmp.ndim == 3:
                    cropper = cube_crop_frames
                else:
                    cropper = frame_crop
            tmp_tmp = cropper(tmp, self.final_sz, self.crop_cen, force=True, verbose=debug)
            write_fits(self.outpath + '2_crop_' + fits_name, tmp_tmp, verbose=debug)

        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath + '2_nan_corr_' + fits_name, verbose=debug)
            tmp_tmp = cropper(tmp, self.final_sz, self.crop_cen, force=True, verbose=debug)
            write_fits(self.outpath + '2_crop_' + fits_name, tmp_tmp, verbose=debug)
        if verbose:
            print('SCI and SKY cubes are cropped to a common size of {} px'.format(self.final_sz), flush=True)

        # Crop the bpix map in a same way
        bpix_map = frame_crop(bpix_map, self.final_sz, cenxy=self.crop_cen, force=True, verbose=debug)
        write_fits(self.outpath + 'master_bpix_map_2ndcrop.fits', bpix_map, verbose=debug)

        # Compare before and after crop
        if plot:
            old_sci = open_fits(self.outpath + '2_nan_corr_' + sci_list[0], verbose=debug)
            crop_sci = open_fits(self.outpath + '2_crop_' + sci_list[0], verbose=debug)
            if old_sci.ndim == 3:
                old_sci = old_sci[-1]
            if crop_sci.ndim == 3:
                crop_sci = crop_sci[-1]
            plot_frames((old_sci, crop_sci, bpix_map), vmin=(np.percentile(old_sci, 0.5), np.percentile(crop_sci, 0.5), 0),
                        vmax=(np.percentile(old_sci, 99.5), np.percentile(crop_sci, 99.5), 1), dpi=300, horsp=0.2,
                        cmap='inferno', top_colorbar=True,
                        label=('Before crop \n' + sci_list[0], 'After crop \n' + sci_list[0], 'Science/Sky Bad Pixel Map \nIdentified using flats'),
                        save=self.outpath + 'Crop_before_bpix_correction.pdf')

        if verbose:
            print('Running bad pixel correction...', flush=True)
            start_time = time_ini(verbose=False)

        # whether to use median to find bad pixels or go frame by frame
        if self.fast_calibration:
            frame_by_frame = False
        else:
            frame_by_frame = True

        def _plot_bpix(file_name1, file_name2, before1, before2, map1, map2, after1, after2, file_type):
            if before1.ndim == 3:
                before1 = before1[0]
            if after1.ndim == 3:
                after1 = after1[0]
            if map1.ndim == 3:
                map1 = map1[0]
            if before2.ndim == 3:
                before2 = before2[0]
            if after2.ndim == 3:
                after2 = after2[0]
            if map2.ndim == 3:
                map2 = map2[0]
            plot_frames((before1, map1, after1, before2, map2, after2), rows=2,
                        vmin=(np.percentile(before1, 0.5), 0, np.percentile(after1, 0.5), np.percentile(before2, 0.5), 0, np.percentile(after2, 0.5)),
                        vmax=(np.percentile(before1, 99.5), 1, np.percentile(after1, 99.5), np.percentile(before2, 99.5), 1, np.percentile(after2, 99.5)),
                        dpi=300, horsp=0.2, cmap='inferno', top_colorbar=True,
                        label=('Before bad pixel correction \n' + file_name1, '{} Bad Pixels Identified'.format(file_type),
                               'After bad pixel correction \n' + file_name1, 'Before bad pixel correction \n' + file_name2,
                               '{} Bad Pixels Identified'.format(file_type), 'After bad pixel correction \n' + file_name2),
                        save=self.outpath + '{}_bpix_correction.pdf'.format(file_type))

        for sc, fits_name in enumerate(sci_list):
            if overwrite or not isfile(self.outpath + '2_bpix_corr_' + fits_name):
                # first with the bp max defined from the flat field (without protecting radius)
                tmp = open_fits(self.outpath + '2_crop_' + fits_name, verbose=debug)
                tmp_tmp = cube_fix_badpix_clump(tmp, bpm_mask=bpix_map, correct_only=True, verbose=debug, nproc=self.nproc)
                write_fits(self.outpath + '2_bpix_corr_' + fits_name, tmp_tmp, verbose=debug)
            # second, residual hot pixels
            if overwrite or not (isfile(self.outpath + '2_bpix_corr2_' + fits_name) and isfile(self.outpath + '2_bpix_corr2_map_' + fits_name)):
                tmp_tmp = open_fits(self.outpath + '2_bpix_corr_' + fits_name, verbose=debug)
                if tmp_tmp.ndim == 3:
                    tmp_tmp = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, size=5,
                                                       protect_mask=10, frame_by_frame=frame_by_frame, verbose=debug,
                                                       nproc=self.nproc)
                else:
                    tmp_tmp = frame_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, size=5,
                                                        protect_mask=10, verbose=debug)
                # create a bpm for the 2nd correction
                tmp_tmp_tmp = tmp_tmp - tmp
                tmp_tmp_tmp = np.where(tmp_tmp_tmp != 0, 1, 0)
                write_fits(self.outpath + '2_bpix_corr2_' + fits_name, tmp_tmp, verbose=debug)
                write_fits(self.outpath + '2_bpix_corr2_map_' + fits_name, tmp_tmp_tmp, verbose=debug)

        if plot:
            before1 = open_fits(self.outpath + '2_crop_' + sci_list[0], verbose=debug)
            map1 = open_fits(self.outpath + '2_bpix_corr2_map_' + sci_list[0], verbose=debug)
            after1 = open_fits(self.outpath + '2_bpix_corr2_' + sci_list[0], verbose=debug)
            before2 = open_fits(self.outpath + '2_crop_' + sci_list[-1], verbose=debug)
            map2 = open_fits(self.outpath + '2_bpix_corr2_map_' + sci_list[-1], verbose=debug)
            after2 = open_fits(self.outpath + '2_bpix_corr2_' + sci_list[-1], verbose=debug)
            _plot_bpix(sci_list[0], sci_list[-1], before1, before2, map1, map2, after1, after2, 'Science')

        if verbose:
            print('*************Bad pixels corrected in SCI cubes*************', flush=True)

        bpix_map = open_fits(self.outpath + 'master_bpix_map_2ndcrop.fits', verbose=debug)
        # t0 = time_ini()
        for sk, fits_name in enumerate(sky_list):
            if overwrite or not isfile(self.outpath + '2_bpix_corr_' + fits_name):
                # first with the bp max defined from the flat field (without protecting radius)
                tmp = open_fits(self.outpath + '2_crop_' + fits_name, verbose=debug)
                tmp_tmp = cube_fix_badpix_clump(tmp, bpm_mask=bpix_map, correct_only=True, verbose=debug, nproc=self.nproc)
                write_fits(self.outpath + '2_bpix_corr_' + fits_name, tmp_tmp, verbose=debug)
            # timing(t0)
            # second, residual hot pixels
            if overwrite or not (isfile(self.outpath + '2_bpix_corr2_' + fits_name) and isfile(self.outpath + '2_bpix_corr2_map_' + fits_name)):
                tmp_tmp = open_fits(self.outpath + '2_bpix_corr_' + fits_name, verbose=debug)
                if tmp_tmp.ndim == 3:
                    tmp_tmp = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, size=5,
                                                       protect_mask=10, frame_by_frame=frame_by_frame, verbose=debug,
                                                       nproc=self.nproc)
                else:
                    tmp_tmp = frame_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, size=5,
                                                        protect_mask=10, verbose=debug)
                # create a bpm for the 2nd correction
                tmp_tmp_tmp = tmp_tmp - tmp
                tmp_tmp_tmp = np.where(tmp_tmp_tmp != 0, 1, 0)
                write_fits(self.outpath + '2_bpix_corr2_' + fits_name, tmp_tmp, verbose=debug)
                write_fits(self.outpath + '2_bpix_corr2_map_' + fits_name, tmp_tmp_tmp, verbose=debug)

        if plot:
            before1 = open_fits(self.outpath + '2_crop_' + sky_list[0], verbose=debug)
            map1 = open_fits(self.outpath + '2_bpix_corr2_map_' + sky_list[0], verbose=debug)
            after1 = open_fits(self.outpath + '2_bpix_corr2_' + sky_list[0], verbose=debug)
            before2 = open_fits(self.outpath + '2_crop_' + sky_list[-1], verbose=debug)
            map2 = open_fits(self.outpath + '2_bpix_corr2_map_' + sky_list[-1], verbose=debug)
            after2 = open_fits(self.outpath + '2_bpix_corr2_' + sky_list[-1], verbose=debug)
            _plot_bpix(sky_list[0], sky_list[-1], before1, before2, map1, map2, after1, after2, 'Sky')

        if verbose:
            print('*************Bad pixels corrected in SKY cubes*************', flush=True)

        if len(unsat_list) > 0:
            bpix_map_unsat = open_fits(self.outpath + 'master_bpix_map_unsat.fits', verbose=debug)
            # t0 = time_ini()
            for un, fits_name in enumerate(unsat_list):
                if overwrite or not isfile(self.outpath + '2_bpix_corr_unsat_' + fits_name):
                    # first with the bp max defined from the flat field (without protecting radius)
                    tmp = open_fits(self.outpath + '2_nan_corr_unsat_' + fits_name, verbose=debug)
                    tmp_tmp = cube_fix_badpix_clump(tmp, bpm_mask=bpix_map_unsat, correct_only=True, verbose=debug, nproc=self.nproc)
                    write_fits(self.outpath + '2_bpix_corr_unsat_' + fits_name, tmp_tmp, verbose=debug)

                if overwrite or not (isfile(self.outpath + '2_bpix_corr2_unsat_' + fits_name) and isfile(self.outpath + '2_bpix_corr2_map_unsat_' + fits_name)):
                    # second, residual hot pixels
                    tmp_tmp = open_fits(self.outpath + '2_bpix_corr_unsat_' + fits_name, verbose=debug)
                    tmp_tmp = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, size=5,
                                                       protect_mask=10, frame_by_frame=frame_by_frame, verbose=debug,
                                                       nproc=self.nproc)
                    # create a bpm for the 2nd correction
                    tmp_tmp_tmp = tmp_tmp - tmp
                    tmp_tmp_tmp = np.where(tmp_tmp_tmp != 0, 1, 0)
                    write_fits(self.outpath + '2_bpix_corr2_unsat_' + fits_name, tmp_tmp, verbose=debug)
                    write_fits(self.outpath + '2_bpix_corr2_map_unsat_' + fits_name, tmp_tmp_tmp, verbose=debug)

            if plot:
                before1 = open_fits(self.outpath + '2_nan_corr_unsat_' + unsat_list[0], verbose=debug)
                map1 = open_fits(self.outpath + '2_bpix_corr2_map_unsat_' + unsat_list[0], verbose=debug)
                after1 = open_fits(self.outpath + '2_bpix_corr2_unsat_' + unsat_list[0], verbose=debug)
                before2 = open_fits(self.outpath + '2_nan_corr_unsat_' + unsat_list[-1], verbose=debug)
                map2 = open_fits(self.outpath + '2_bpix_corr2_map_unsat_' + unsat_list[-1], verbose=debug)
                after2 = open_fits(self.outpath + '2_bpix_corr2_unsat_' + unsat_list[-1], verbose=debug)
                _plot_bpix(unsat_list[0], unsat_list[-1], before1, before2, map1, map2, after1, after2, 'Unsat')

            if verbose:
                print('*************Bad pixels corrected in UNSAT cubes*************', flush=True)

        if remove:
            system("rm " + self.outpath + '2_nan_corr_*')
            system("rm " + self.outpath + '2_crop_*')

        if verbose:
            timing(start_time)

    def first_frames_removal(self, nrm='auto', verbose=True, debug=False, plot=True, remove=False):
        """
        Corrects for the inconsistent DIT times within NACO cubes.
        The first few frames are removed and the rest rescaled such that the flux is constant.

        nrm: 'auto' or int, optional
            Number of frames to remove (nrm) at the start of each cube.
            If 'auto', the pipeline determines the number of frames to remove automatically
            by measuring the flux of the star.
        verbose: bool, optional
            Print relevant information.
        debug: bool, optional
            Prints significantly more information.
        plot: bool, optional
            Save relevant plot showing flux vs frame
        remove: bool, optional
            Removes redundant files.

        """
        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])
        n_sci = len(sci_list)

        if not isfile(self.outpath + '2_bpix_corr2_' + sci_list[-1]):
            raise NameError('Missing 2_bpix_corr2_*.fits. Run: correct_bad_pixels()')

        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])
        n_sky = len(sky_list)

        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        self.final_sz = int(open_fits(self.outpath + 'final_sz', verbose=debug)[0])

        com_sz = open_fits(self.outpath + '2_bpix_corr2_' + sci_list[0], verbose=debug).shape[-1]
        # obtaining the real ndit values of the frames (not that of the headers)
        tmp = np.zeros([n_sci, com_sz, com_sz])
        self.real_ndit_sci = []
        for sc, fits_name in enumerate(sci_list):
            tmp_tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            if tmp_tmp.ndim == 3:
                tmp[sc] = tmp_tmp[-1]
                self.real_ndit_sci.append(tmp_tmp.shape[0] - 1)
            else:
                tmp[sc] = tmp_tmp.copy()
                self.real_ndit_sci.append(1)

        tmp = np.zeros([n_sky, com_sz, com_sz])
        self.real_ndit_sky = []
        for sk, fits_name in enumerate(sky_list):
            tmp_tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            if tmp_tmp.ndim == 3:
                tmp[sk] = tmp_tmp[-1]
                self.real_ndit_sky.append(tmp_tmp.shape[0] - 1)
            else:
                tmp[sk] = tmp_tmp.copy()
                self.real_ndit_sky.append(1)

        min_ndit_sci = int(np.amin(self.real_ndit_sci))

        write_fits(self.outpath + 'real_ndit_sci.fits', np.array(self.real_ndit_sci), verbose=debug)
        write_fits(self.outpath + 'real_ndit_sky.fits', np.array(self.real_ndit_sky), verbose=debug)

        if verbose:
            print("Real number of frames in each SCI cube = ", self.real_ndit_sci)
            print("Real number of frames in each SKY cube = ", self.real_ndit_sky)
            print("Nominal frames: {}, min frames when skimming through cubes: {}".format(self.dataset_dict['ndit_sci'],
                                                                                      min_ndit_sci), flush=True)

        # update the final size and subsequently the mask
        mask_inner_rad = int(3.0 / self.dataset_dict['pixel_scale'])
        mask_width = int((self.final_sz / 2.) - mask_inner_rad - 2)

        # measure the flux in sci avoiding the star at the centre (3'' should be enough)
        tmp_fluxes = np.zeros([n_sci, min_ndit_sci])
        bar = pyprind.ProgBar(n_sci, stream=1, title='Estimating flux in SCI frames')
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            for ii in range(min_ndit_sci):
                tmp_tmp = get_annulus_segments(tmp[ii], mask_inner_rad, mask_width, mode='mask')[0]
                tmp_fluxes[sc, ii] = np.sum(tmp_tmp)
            bar.update()
        tmp_flux_med = np.median(tmp_fluxes, axis=0)
        if verbose:
            print('Flux in SCI frames has been measured', flush=True)

        # create a plot of the median flux in the frames
        med_flux = np.median(tmp_flux_med)
        std_flux = np.std(tmp_flux_med)
        if verbose:
            print("Median flux: {}".format(med_flux))
            print("Standard deviation of flux: {}".format(std_flux), flush=True)

        if nrm == 'auto':
            first_time = True
            good = []
            good_flux = []
            bad = []
            bad_flux = []
            for ii in range(min_ndit_sci):
                if tmp_flux_med[ii] > med_flux + (2 * std_flux) or tmp_flux_med[ii] < med_flux - (2 * std_flux) or ii == 0:
                    bad_flux.append(tmp_flux_med[ii] / med_flux)
                    bad.append(ii + 1)
                else:
                    good_flux.append(tmp_flux_med[ii] / med_flux)
                    good.append(ii + 1)
                    if first_time:
                        nfr_rm = ii  # the ideal number is when the flux is within 3 standard deviations
                        nfr_rm = min(nfr_rm, 10)  # if above 10 frames to remove, it will cap nfr_rm to 10
                        first_time = False

            if plot:
                plt.close('all')
                plt.plot(good, good_flux, 'bo', label='Good')
                plt.plot(bad, bad_flux, 'ro', label='Bad')
                plt.title('Flux in SCI frames')
                plt.ylabel('Normalised flux')
                plt.xlabel('Frame number')
                plt.legend()
                plt.grid(alpha=0.2)
                plt.savefig(self.outpath + "Variability_of_DIT.pdf", bbox_inches='tight', pad_inches=0.1)
                plt.close('all')
        else:
            nfr_rm = nrm  # if the number of frames to remove is manually provided

        if verbose:
            print("The number of frames to remove at the beginning is: ", nfr_rm, flush=True)

        # update the range of frames that will be cut off.
        for zz in range(len(self.real_ndit_sci)):
            self.real_ndit_sci[zz] = min(self.real_ndit_sci[zz] - nfr_rm, min(self.dataset_dict['ndit_sci']) - nfr_rm)
        min_ndit_sky = min(self.real_ndit_sky)
        for zz in range(len(self.real_ndit_sky)):
            self.real_ndit_sky[zz] = min_ndit_sky - nfr_rm

        self.new_ndit_sci = min(self.dataset_dict['ndit_sci']) - nfr_rm
        self.new_ndit_sky = min(self.dataset_dict['ndit_sky']) - nfr_rm
        self.new_ndit_unsat = min(self.dataset_dict['ndit_unsat']) - nfr_rm

        write_fits(self.outpath + 'new_ndit_sci_sky_unsat.fits',
                   np.array([self.new_ndit_sci, self.new_ndit_sky, self.new_ndit_unsat]), verbose=debug)

        if verbose:
            print("The new number of frames in each SCI cube is: ", self.new_ndit_sci)
            print("The new number of frames in each SKY cube is: ", self.new_ndit_sky)
            print("The new number of frames in each UNSAT cube is: ", self.new_ndit_unsat, flush=True)

        if isfile(self.inpath + "derot_angles_uncropped.fits"):
            angles = open_fits(self.inpath + "derot_angles_uncropped.fits", verbose=debug)
            angles = angles[:, nfr_rm:]  # crops each cube of rotation angles file, by keeping all cubes but removing the number of frames at the start
            write_fits(self.outpath + 'derot_angles_cropped.fits', angles, verbose=debug)
        else:
            raise NameError('Missing derot_angles_uncropped.fits')

        # Actual cropping of the cubes to remove the first frames, and the last one (median) AND RESCALING IN FLUX
        # FOR SCI
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            if tmp.ndim == 3:
                tmp_tmp = np.zeros([int(self.real_ndit_sci[sc]), tmp.shape[1], tmp.shape[2]])
                for dd in range(nfr_rm, nfr_rm + int(self.real_ndit_sci[sc])):
                    tmp_tmp[dd - nfr_rm] = tmp[dd] * np.median(tmp_fluxes[sc]) / tmp_fluxes[sc, dd]
            else:
                tmp_tmp = tmp.copy()
            write_fits(self.outpath + '3_rmfr_' + fits_name, tmp_tmp, verbose=debug)

        if verbose:
            print('The first {} frames were removed and the flux rescaled for SCI cubes'.format(nfr_rm), flush=True)

        # FOR SKY
        tmp_fluxes_sky = np.zeros([n_sky, self.new_ndit_sky])
        bar = pyprind.ProgBar(n_sky, stream=1, title='Estimating flux in SKY frames')
        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            for ii in range(nfr_rm, nfr_rm + self.new_ndit_sky):
                tmp_tmp = get_annulus_segments(tmp[ii], mask_inner_rad, mask_width, mode='mask')[0]
                tmp_fluxes_sky[sk, ii - nfr_rm] = np.sum(tmp_tmp)
            bar.update()

        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath + '2_bpix_corr2_' + fits_name, verbose=debug)
            tmp_tmp = np.zeros([int(self.real_ndit_sky[sk]), tmp.shape[1], tmp.shape[2]])
            for dd in range(nfr_rm, nfr_rm + int(self.real_ndit_sky[sk])):
                tmp_tmp[dd - nfr_rm] = tmp[dd] * np.median(tmp_fluxes_sky[sk, nfr_rm:]) / tmp_fluxes_sky[sk, dd - nfr_rm]
            write_fits(self.outpath + '3_rmfr_' + fits_name, tmp_tmp, verbose=debug)

        if verbose:
            print('The first {} frames were removed and the flux rescaled for SKY cubes'.format(nfr_rm), flush=True)

        # Science master cube for identifying dust specks later on
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
            if sc == 0:
                cube_meds = np.zeros([n_sci, tmp.shape[-2], tmp.shape[-1]])
            if tmp.ndim == 3:
                cube_meds[sc] = np.median(tmp, axis=0)
            elif tmp.ndim == 2:
                cube_meds[sc] = tmp.copy()
            else:
                raise ValueError('A science cube has dimensions outside the expected range.')
        write_fits(self.outpath + "TMP_med_bef_SKY_subtr.fits", np.median(cube_meds, axis=0), verbose=debug)
            
        if len(unsat_list) > 0:
            for un, fits_name in enumerate(unsat_list):
                tmp = open_fits(self.outpath + '2_bpix_corr2_unsat_' + fits_name, verbose=debug)
                tmp_tmp = tmp[nfr_rm:-1]
                write_fits(self.outpath + '3_rmfr_unsat_' + fits_name, tmp_tmp, verbose=debug)
            if verbose:
                print('The first {} frames were removed from the UNSAT cubes'.format(nfr_rm), flush=True)

        if remove:
            system("rm " + self.outpath + '2_bpix_corr_*')
            system("rm " + self.outpath + '2_bpix_corr2_*')
            system("rm " + self.outpath + '2_bpix_corr2_map_*')
            system("rm " + self.outpath + '2_bpix_corr_unsat_*')
            system("rm " + self.outpath + '2_bpix_corr2_unsat_*')
            system("rm " + self.outpath + '2_bpix_corr2_map_unsat_*')

    def get_stellar_psf(self, nd_filter=False, plot=True, verbose=True, debug=False, remove=False):
        """
        Obtain a PSF model of the star based off of the unsaturated cubes.

        For each unsaturated cube, the star is imaged without the use of a coronagraphic mask. Generally, the star is
        also 'dithered' from one quadrant of the detector to another, avoiding the bottom left quadrant. This function
        first median combines all unsaturated frames into a single frame, and subtracts it from the median of each cube.
        As the star is dithered, it is not subtracted through this process and is easy to identify in the sky subtracted
        image. We then apply aperture photometry to ensure the flux of the star for each unique position is within
        expectations wrt. each other. Finally, we apply a 1 arcsecond threshold on the separation of the star between
        each dithered cube. This process aims to prevent background stars or artefacts being detected as the star. Once
        this process is complete, we crop to a small area around the star, create a master cube, recenter all frames
        using a 2D Gaussian, and stack to create a final point spread function. If observing conditions are bad, some
        frames will be trimmed using a Pearson correlation to the master frame. A neutral density filter toggle is
        available for unsaturated cubes taken with a neutral density filter applied.

        Should be improved to handle the case of no dithering.

        nd_filter : bool, default: None
            when a ND filter is used in L' the transmission is ~0.01777. Used for scaling
        plot : bool, optional
            Save relevant plots to the outpath as PDF.
        verbose and debug : bool
            prints more info, if debug it prints when files are opened and gives some additional info.
            verbose is on by default.
        remove options : bool, False by default
            Cleans previous calibration files
        """
        start_time = time_ini(verbose=False)
        unsat_list = []
        with open(self.inpath + "unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])
        if not isfile(self.outpath + '3_rmfr_unsat_' + unsat_list[-1]):
            raise NameError('Missing 3_rmfr_unsat*.fits. Run: first_frame_removal()')

        # get the new number of frames in each cube after trimming
        self.new_ndit_unsat = int(open_fits(self.outpath + 'new_ndit_sci_sky_unsat', verbose=debug)[2])
        self.resel_ori = self.dataset_dict['wavelength'] * 206265 / (
                self.dataset_dict['size_telescope'] * self.dataset_dict['pixel_scale'])

        if verbose:
            print('Unsaturated cubes list:', unsat_list)
            print('Number of frames in each cube:', self.new_ndit_unsat)
            print('Resolution element: {} px'.format(self.resel_ori), flush=True)

        # open first unsaturated cube, get size, make empty array for storing median of each cube
        # this does assume the unsaturated cubes have three dimensions, which they always should
        tmp = open_fits(self.outpath + '3_rmfr_unsat_' + unsat_list[0], verbose=debug)
        tmp_med = np.zeros([len(unsat_list), tmp.shape[-1], tmp.shape[-1]])
        for ii, fits_name in enumerate(unsat_list):
            tmp_med[ii] = np.nanmedian(open_fits(self.outpath + '3_rmfr_unsat_' + fits_name, verbose=debug), axis=0)
        tmp_med = np.nanmedian(tmp_med, axis=0)  # median frame of median cube
        write_fits(self.outpath + 'median_unsat.fits', tmp_med, verbose=debug)

        unsat_pos = []
        # obtain star positions in the sky subtracted (median unsat minus unsat) unsaturated frames
        # could be made faster by using the cube already made in lines above
        for fits_name in unsat_list:
            unsat = np.nanmedian(open_fits(self.outpath + '3_rmfr_unsat_' + fits_name, verbose=debug), axis=0)
            unsat_sky = unsat - tmp_med  # sky subtraction for unsaturated frames
            tmp = find_filtered_max(unsat_sky)  # get star location
            unsat_pos.append(tmp)

        if verbose:
            print('Position [y,x] of the star in unsaturated cubes:', unsat_pos, flush=True)

        flux_list = []
        # Measure the flux at those positions
        for un, fits_name in enumerate(unsat_list):
            if un == 0:
                print('Aperture radius is {} px'.format(round(3 * self.resel_ori)), flush=True)
            circ_aper = CircularAperture(positions=(unsat_pos[un][1], unsat_pos[un][0]), r=round(3 * self.resel_ori))
            tmp = open_fits(self.outpath + '3_rmfr_unsat_' + fits_name, verbose=debug)
            tmp = np.median(tmp, axis=0)
            circ_aper_phot = aperture_photometry(tmp, circ_aper, method='exact')
            circ_flux = np.array(circ_aper_phot['aperture_sum'])
            flux_list.append(circ_flux[0])

        med_flux = np.median(flux_list)
        std_flux = np.std(flux_list)

        if verbose:
            print('Fluxes in unsaturated frames: {} adu'.format(flux_list))
            print('Median flux and standard deviations:', med_flux, std_flux, flush=True)

        good_unsat_list = []
        good_unsat_pos = []
        # define good unsat list where the flux of the stars is within 3 standard devs
        for i, flux in enumerate(flux_list):
            if med_flux + 3 * std_flux > flux > med_flux - 3 * std_flux:
                good_unsat_list.append(unsat_list[i])
                good_unsat_pos.append(unsat_pos[i])

        if verbose:
            print('Good unsaturated cubes:', good_unsat_list)
            print('Good unsaturated positions:', good_unsat_pos, flush=True)

        unsat_mjd_list = []
        # get times of unsat cubes (modified Julian calendar)
        for fname in unsat_list:
            tmp, header = open_fits(self.inpath + fname, header=True, verbose=debug)
            unsat_mjd_list.append(header['MJD-OBS'])

        thr_d = (1.0 / self.dataset_dict[
            'pixel_scale'])  # threshold: difference in star pos must be greater than 1 arc sec
        if verbose:
            print('Unsaturated cube observation time (MJD):', unsat_mjd_list)
            print('Distance threshold: {:.3f} px'.format(thr_d), flush=True)

        index_dither = [0]  # first position to start measuring offset/dithering distance
        unique_pos = [unsat_pos[0]]  # we know the first position is unique, so its first entry of unique positions
        counter = 1
        for un, pos in enumerate(unsat_pos[1:]):  # looks at all positions after the first one
            new_pos = True
            for i, uni_pos in enumerate(unique_pos):
                if dist(int(pos[1]), int(pos[0]), int(uni_pos[1]), int(uni_pos[0])) < thr_d:
                    index_dither.append(i)
                    new_pos = False
                    break
            if new_pos:
                unique_pos.append(pos)
                index_dither.append(counter)
                counter += 1

        all_idx = [i for i in range(len(unsat_list))]

        if verbose:
            print('Unique positions:', unique_pos)
            print('Index of all cubes:', all_idx)
            print('Index of dithered cubes:', index_dither)
            print('Starting sky subtraction using cubes on different parts of the detector', flush=True)

        for un, fits_name in enumerate(unsat_list):
            if fits_name in good_unsat_list:  # just consider the good ones
                tmp = open_fits(self.outpath + '3_rmfr_unsat_' + fits_name, verbose=debug)
                # index of cubes on a different part of the detector
                good_idx = [j for j in all_idx if index_dither[j] != index_dither[un]]
                best_idx = find_nearest([unsat_mjd_list[i] for i in good_idx], unsat_mjd_list[un], output='index')
                # best_idx = find_nearest(unsat_mjd_list[good_idx[0]:good_idx[-1]],unsat_mjd_list[un])
                if verbose:
                    print('Index of cubes on different part of detector:', good_idx)
                    print('Index of the above selected for sky subtraction:', best_idx, flush=True)
                tmp_sky = np.median(open_fits(self.outpath + '3_rmfr_unsat_' + unsat_list[good_idx[best_idx]],
                                              verbose=debug), axis=0)
                write_fits(self.outpath + '4_sky_subtr_unsat_' + unsat_list[un], tmp - tmp_sky, verbose=debug)

        if plot:
            # plot cubes before and after sky subtraction, with original on left and sky subtracted on right
            unsat_and_sky_cube = []
            for i, fname in enumerate(good_unsat_list):
                unsat_and_sky_cube.append(np.median(open_fits(self.outpath + '3_rmfr_unsat_' + fname, verbose=debug),
                                                    axis=0))
                unsat_and_sky_cube.append(np.median(open_fits(self.outpath + '4_sky_subtr_unsat_' + fname,
                                                              verbose=debug), axis=0))
            unsat_and_sky_cube = tuple(unsat_and_sky_cube)  # plot_frames only takes tuples of frames

            # create labels using the names of each cube, have them alternate so they align with each frame in the plot
            labels = tuple(x for y in zip(['3_rmfr_unsat_' + s for s in good_unsat_list],
                                          ['4_sky_subtr_unsat_' + s for s in good_unsat_list]) for x in y)

            plot_frames(unsat_and_sky_cube, rows=len(good_unsat_list), title='Unsaturated cube sky subtraction',
                        label=labels, label_size=8, cmap='inferno', dpi=300,
                        save=self.outpath + 'Unsat_skysubtracted.pdf')
            plt.close('all')

        if remove:
            for un, fits_name in enumerate(unsat_list):
                system("rm " + self.outpath + '3_rmfr_unsat_' + fits_name)

        crop_sz_tmp = int(8 * self.resel_ori)
        crop_sz = int(7 * self.resel_ori)
        if not crop_sz_tmp % 2:  # if it's not even, crop
            crop_sz_tmp += 1
        if not crop_sz % 2:
            crop_sz += 1
        # psf_tmp = np.zeros([len(good_unsat_list) * self.new_ndit_unsat, crop_sz, crop_sz])
        psf_tmp = []
        #successful_unsat_idx = []
        for un, fits_name in enumerate(good_unsat_list):
            tmp = open_fits(self.outpath + '4_sky_subtr_unsat_' + fits_name, verbose=debug)
            xy = (good_unsat_pos[un][1], good_unsat_pos[un][0])
            tmp_tmp, _, _ = cube_crop_frames(tmp, crop_sz_tmp, xy=xy, verbose=debug, full_output=True)
            cy, cx = frame_center(tmp_tmp[0], verbose=debug)
            write_fits(self.outpath + '4_tmp_crop_' + fits_name, tmp_tmp, verbose=debug)
            try:
                tmp_tmp = cube_recenter_2dfit(tmp_tmp, xy=(int(cx), int(cy)), fwhm=self.resel_ori, subi_size=9,
                                              nproc=self.nproc, model='gauss', full_output=False, verbose=debug,
                                              save_shifts=False, offset=None, negative=False, debug=False,
                                              threshold=False, plot=False)
                tmp_tmp = cube_crop_frames(tmp_tmp, crop_sz, xy=(cx, cy), verbose=debug)
                write_fits(self.outpath + '4_centered_unsat_' + fits_name, tmp_tmp, verbose=debug)
                for dd in range(self.new_ndit_unsat):
                    # combining all frames in unsat to make master cube
                    # psf_tmp[un * self.new_ndit_unsat + dd] = tmp_tmp[dd]
                    psf_tmp.append(tmp_tmp[dd])
                #successful_unsat_idx.append(un)
            except:
                msg = 'Warning: Unable to fit to unsaturated frame 4_sky_subtr_unsat_{}\nIt is safe to continue, ' \
                      'but you may want to check the final unsaturated master cube'.format(fits_name)
                print(msg, flush=True)
        #psf_tmp = psf_tmp[successful_unsat_idx]
        psf_tmp = np.array(psf_tmp)
        write_fits(self.outpath + 'tmp_master_unsat_psf.fits', psf_tmp, verbose=debug)

        good_unsat_idx, bad_unsat_list = cube_detect_badfr_pxstats(psf_tmp, mode='circle', in_radius=5, top_sigma=1,
                                                                   low_sigma=0.2, window=10, plot=True, verbose=verbose)
        if plot:
            plt.savefig(self.outpath + 'unsat_bad_frame_detection.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.close('all')
        # use only the good frames, median combine and save
        psf_tmp = psf_tmp[good_unsat_idx]
        write_fits(self.outpath + 'tmp_master_unsat_psf_trimmed.fits', psf_tmp, verbose=debug)
        psf_med = np.median(psf_tmp, axis=0)
        write_fits(self.outpath + 'master_unsat_psf.fits', psf_med, verbose=debug)

        # calculates the correlation of each frame to the median and saves as a list
        # distances = cube_distance(psf_tmp, psf_med, mode='full', dist='pearson', plot=plot)
        # if plot:  # save a plot of distances compared to the median for each frame if set to 'save'
        #     plt.savefig(self.outpath + 'distances_unsat.pdf', format='pdf')
        #     plt.close('all')
        #
        # # threshold is the median of the distances minus half a stddev
        # correlation_thres = np.median(distances) - 0.5 * np.std(distances)
        # # detect and remove bad frames in a subframe two pixels smaller than original frame
        # good_frames, _ = cube_detect_badfr_correlation(psf_tmp, crop_size=psf_med.shape[-1] - 2, frame_ref=psf_med,
        #                                                dist='pearson', threshold=correlation_thres, plot=plot,
        #                                                verbose=verbose)
        #
        # if plot:
        #     plt.savefig(self.outpath + 'frame_correlation_unsat.pdf', format='pdf')
        #     plt.close('all')
        # # select only the good frames and median combine
        # psf_tmp = psf_tmp[good_frames]
        # write_fits(self.outpath + 'tmp_master_unsat_psf_trimmed.fits', psf_tmp, verbose=debug)
        # psf_med = np.median(psf_tmp, axis=0)
        # write_fits(self.outpath + 'master_unsat_psf.fits', psf_med, verbose=debug)
        #
        # ### second round
        # distances = cube_distance(psf_tmp, psf_med, mode='full', dist='pearson', plot=plot)
        # if plot:
        #     plt.savefig(self.outpath + 'distances_unsat2.pdf', format='pdf')
        #     plt.close('all')
        #
        # # threshold is the median of the distances minus one stddev
        # correlation_thres = np.median(distances) - np.std(distances)
        # good_frames, _ = cube_detect_badfr_correlation(psf_tmp, crop_size=psf_med.shape[-1] - 2, frame_ref=psf_med,
        #                                                dist='pearson', threshold=correlation_thres, plot=plot,
        #                                                verbose=verbose)
        #
        # if plot:
        #     plt.savefig(self.outpath + 'frame_correlation_unsat2.pdf', format='pdf')
        #     plt.close('all')
        # psf_tmp = psf_tmp[good_frames]
        # write_fits(self.outpath + 'tmp_master_unsat_psf_trimmed2.fits', psf_tmp, verbose=debug)
        # psf_med = np.median(psf_tmp, axis=0)
        # write_fits(self.outpath + 'master_unsat_psf.fits', psf_med, verbose=debug)

        if verbose:
            print('The median PSF of the star has been obtained', flush=True)
        if plot:
            plot_frames(psf_med, dpi=300, label='Median PSF', vmin=np.percentile(psf_med, 0.1), vmax=np.percentile(psf_med, 99.9),
                        cmap='inferno', colorbar_label='Flux [adu per {}s]'.format(self.dit_unsat), log=True,
                        save=self.outpath + 'Median_PSF.pdf')
            plt.close('all')

        data_frame = fit_2dgaussian(psf_med, crop=False, cent=None, fwhmx=self.resel_ori, fwhmy=self.resel_ori, theta=0,
                                    threshold=False, sigfactor=6, full_output=True, debug=plot)
        if plot:  # saves the model
            plt.savefig(self.outpath + 'PSF_fitting.pdf')
            plt.close('all')

        data_frame = data_frame.astype('float64')
        self.fwhm_y = data_frame['fwhm_y'][0]
        self.fwhm_x = data_frame['fwhm_x'][0]
        self.fwhm_theta = data_frame['theta'][0]
        self.fwhm = (self.fwhm_y + self.fwhm_x) / 2.0

        if verbose:
            print("fwhm_y, fwhm x, theta and fwhm (mean of both):")
            print(self.fwhm_y, self.fwhm_x, self.fwhm_theta, self.fwhm, flush=True)
        write_fits(self.outpath + 'fwhm.fits', np.array([self.fwhm, self.fwhm_y, self.fwhm_x, self.fwhm_theta]),
                   verbose=debug)

        psf_med_norm, flux_unsat, _ = normalize_psf(psf_med, fwhm=self.fwhm, full_output=True)
        if nd_filter:
            print('Neutral Density filter toggle is on... using a transmission of 1.78238% for 3.78 micrometers', flush=True)
            flux_psf = (flux_unsat[0] * (1 / 0.0178238)) * (self.dataset_dict['dit_sci'] / self.dataset_dict['dit_unsat'])
            # scales flux by DIT ratio accounting for transmission of ND filter
        else:
            flux_psf = flux_unsat[0] * (self.dataset_dict['dit_sci'] / self.dataset_dict['dit_unsat'])
            # scales flux by DIT ratio

        write_fits(self.outpath + 'master_unsat_psf_norm.fits', psf_med_norm, verbose=debug)
        write_fits(self.outpath + 'master_unsat-stellarpsf_fluxes.fits', np.array([flux_unsat[0], flux_psf]),
                   verbose=debug)

        if verbose:
            print("Flux of the PSF (scaled for science frames): ", flux_psf)
            print("FWHM = {} px".format(round(self.fwhm, 3)), flush=True)
        timing(start_time)
        if self.fwhm < 3 or self.fwhm > 6:
            raise ValueError('FWHM is not within expected values!')

    def subtract_sky(self, imlib='vip-fft', npc=1, mode='PCA', fwhm=None, 
                     verbose=True, debug=False, plot=None, remove=False):
        """
        Sky subtraction of the science cubes
        imlib : string: 'ndimage-interp', 'opencv', 'vip-fft'
        npc : list, None, integer
        mode : string: 'PCA', 'median'
        fwhm : float or None. If not None, supersedes value from get_stellar_psf
        plot : Save relevant plots
        remove options: True, False. Cleans file for unused fits
        """
        sky_list = []
        with open(self.inpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])
        n_sky = len(sky_list)

        sci_list = []
        with open(self.inpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])
        n_sci = len(sci_list)

        # save sci_list.txt to outpath to be used in preproc
        with open(self.outpath + "sci_list.txt", "w") as f:
            for sci in sci_list:
                f.write(sci + '\n')

        if fwhm is None and isfile(self.outpath + 'fwhm.fits'):
            self.fwhm = open_fits(self.outpath + 'fwhm.fits', verbose=debug)[0]
        elif fwhm is not None:
            self.fwhm = fwhm
        elif fwhm is not None:
            raise NameError('FWHM of the star is not defined nor provided. Run: get_stellar_psf()')

        if not isfile(self.outpath + '3_rmfr_' + sci_list[-1]):
            raise NameError('Missing 3_rmfr_*.fits. Run: first_frame_removal()')

        self.final_sz = int(open_fits(self.outpath + 'final_sz', verbose=debug)[
                                0])  # just a single integer in this file to set as final_sz
        self.com_sz = int(open_fits(self.outpath + 'common_sz', verbose=debug)[
                              0])  # just a single integer in this file to set as com_sz

        self.real_ndit_sky = []
        for sk, fits_name in enumerate(sky_list):
            tmp_cube = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
            self.real_ndit_sky.append(tmp_cube.shape[0])

        self.new_ndit_sci = int(open_fits(self.outpath + 'new_ndit_sci_sky_unsat', verbose=debug)[
                                    0])  # the new dimension of the unsaturated sci cube is the first entry
        self.new_ndit_sky = int(open_fits(self.outpath + 'new_ndit_sci_sky_unsat', verbose=debug)[
                                    1])  # the new dimension of the unsaturated sky cube is the second entry
        # self.real_ndit_sky = int(open_fits(self.outpath + 'real_ndit_sky.fits')[0]) # i have a feeling this line doesn't need to exist since it's all saved with self
        #        with open(self.outpath+"real_ndit_sky_list.txt", "r") as f:
        #            tmp = f.readlines()
        #            for line in tmp:
        #                self.real_ndit_sky.append(int(line.split('\n')[0]))
        # pdb.set_trace()
        sky_list_mjd = []
        # get times of sky cubes (modified jullian calander)
        for fname in sky_list:
            tmp, header = open_fits(self.inpath + fname, header=True, verbose=debug)
            sky_list_mjd.append(header['MJD-OBS'])

        # SORT SKY_LIST in chronological order (important for calibration)
        arg_order = np.argsort(sky_list_mjd, axis=0)
        myorder = arg_order.tolist()
        sorted_sky_list = [sky_list[i] for i in myorder]
        sorted_sky_mjd_list = [sky_list_mjd[i] for i in myorder]
        sky_list = sorted_sky_list
        sky_mjd_list = np.array(sorted_sky_mjd_list)
        write_fits(self.outpath + "sky_mjd_times.fits", sky_mjd_list, verbose=debug)

        tmp = open_fits(self.outpath + "TMP_med_bef_SKY_subtr.fits", verbose=debug)
        
        # REALIGNMENT in case of coronagraphic observations
        if self.coro:
            # try high pass filter to isolate blobs
            hpf_sz = int(2 * self.fwhm)
            if not hpf_sz % 2:
                hpf_sz += 1
            tmp = frame_filter_highpass(tmp, mode='median-subt', median_size=hpf_sz,
                                        kernel_size=hpf_sz, fwhm_size=self.fwhm)

            if plot:
                plot_frames(tmp, title='Isolated dust grains', vmax=np.percentile(tmp, 99.9), vmin=np.percentile(tmp, 0.1),
                            dpi=300, save=self.outpath + 'Isolated_grains.pdf')
            # then use the automatic detection tool
            snr_thr = 10
            snr_thr_all = 30
            psfn = open_fits(self.outpath + "master_unsat_psf_norm.fits", verbose=debug)
            table_det = detection(tmp, psf=psfn, bkg_sigma=1, mode='lpeaks', matched_filter=True,
                                  mask=True, snr_thresh=snr_thr, plot=False, debug=False,
                                  full_output=True, verbose=debug, nproc=self.nproc)
            y_dust = table_det['y']
            x_dust = table_det['x']
            snr_dust = table_det['px_snr']
    
            # trim to just keep the specks with SNR>10 anywhere but in the lower left quadrant
            dust_xy_all = []
            dust_xy_tmp = []
            cy, cx = frame_center(tmp)
            for i in range(len(y_dust)):
                if not np.isnan(snr_dust[i]):  # discard nan
                    if abs(y_dust[i] - cy) > 3 * self.fwhm and abs(x_dust[i] - cx) > 3 * self.fwhm:
                        if snr_dust[i] > snr_thr_all:
                            dust_xy_all.append((x_dust[i], y_dust[i]))
                        if (y_dust[i] > cy or x_dust[i] > cx) and snr_dust[i] > snr_thr:  # discard lower left quadrant
                            dust_xy_tmp.append((x_dust[i], y_dust[i]))
            ndust_all = len(dust_xy_all)
    
            ndust = len(dust_xy_tmp)
            if verbose:
                print(dust_xy_tmp)
                print("{} dust specks have been identified for alignment of SCI and SKY frames".format(ndust), flush=True)
    
            # Fit them to gaussians in a test frame, and discard non-circular one (fwhm_y not within 20% of fwhm_x)
    
            test_xy = np.zeros([ndust, 2])
            fwhm_xy = np.zeros([ndust, 2])
            tmp = open_fits(self.outpath + "TMP_med_bef_SKY_subtr.fits", verbose=debug)
            tmp = frame_filter_highpass(tmp, mode='median-subt', median_size=hpf_sz,
                                        kernel_size=hpf_sz, fwhm_size=self.fwhm)
            bad_dust = []
            crop_sz = int(5 * self.resel_ori)
            if crop_sz % 2 == 0:
                crop_sz = crop_sz - 1
    
            for dd in range(ndust):
                table_gaus = fit_2dgaussian(tmp, crop=True, cent=dust_xy_tmp[dd],
                                            cropsize=crop_sz, fwhmx=self.resel_ori,
                                            threshold=True, sigfactor=0,
                                            full_output=True, debug=False)
                test_xy[dd, 1] = table_gaus['centroid_y'][0]
                test_xy[dd, 0] = table_gaus['centroid_x'][0]
                fwhm_xy[dd, 1] = table_gaus['fwhm_y'][0]
                fwhm_xy[dd, 0] = table_gaus['fwhm_x'][0]
                amplitude = table_gaus['amplitude'][0]
                if fwhm_xy[dd, 1] / fwhm_xy[dd, 0] < 0.8 or fwhm_xy[dd, 1] / fwhm_xy[dd, 0] > 1.2:
                    bad_dust.append(dd)
    
            dust_xy = [xy for i, xy in enumerate(dust_xy_tmp) if i not in bad_dust]
            ndust = len(dust_xy)
            if verbose:
                print("We detected {:.0f} non-circular dust specks, hence removed from the list.".format(len(bad_dust)))
                print("We are left with {:.0f} dust specks for alignment of SCI and SKY frames.".format(ndust), flush=True)
    
            # the code first finds the exact coords of the dust features in the median of the first SCI cube (and show them)
            xy_cube0 = np.zeros([ndust, 2])
            crop_sz = int(3 * self.resel_ori)
            tmp_cube = open_fits(self.outpath + '3_rmfr_' + sci_list[0], verbose=debug)
            tmp_med = np.median(tmp_cube, axis=0)
            tmp = frame_filter_highpass(tmp_med, mode='median-subt', median_size=hpf_sz,
                                        kernel_size=hpf_sz, fwhm_size=self.fwhm)
            for dd in range(ndust):
                try:
                    df = fit_2dgaussian(tmp, crop=True, cent=dust_xy[dd], cropsize=crop_sz, fwhmx=self.resel_ori,
                                        fwhmy=self.resel_ori,
                                        theta=0, threshold=True, sigfactor=0, full_output=True,
                                        debug=False)
                    xy_cube0[dd, 1] = df['centroid_y'][0]
                    xy_cube0[dd, 0] = df['centroid_x'][0]
                    fwhm_y = df['fwhm_y'][0]
                    fwhm_x = df['fwhm_x'][0]
                    amplitude = df['amplitude'][0]
                    if verbose:
                        print("coord_x: {}, coord_y: {}, fwhm_x: {}, fwhm_y:{}, amplitude: {}".format(xy_cube0[dd, 0],
                                                                                                      xy_cube0[dd, 1],
                                                                                                      fwhm_x, fwhm_y,
                                                                                                      amplitude), flush=True)
                    shift_xy_dd = (xy_cube0[dd, 0] - dust_xy[dd][0], xy_cube0[dd, 1] - dust_xy[dd][1])
                    if verbose:
                        print("shift with respect to center for dust grain #{}: {}".format(dd, shift_xy_dd), flush=True)
                except ValueError:
                    xy_cube0[dd, 0], xy_cube0[dd, 1] = dust_xy[dd]
                    print("!!! Gaussian fit failed for dd = {}. We set position to first (eye-)guess position.".format(dd), flush=True)
            if verbose:
                print("Note: the shifts should be small.", flush=True)
    
            # then it finds the centroids in all other frames (SCI+SKY) to determine the relative shifts to be applied to align all frames
            shifts_xy_sci = np.zeros([ndust, n_sci, self.new_ndit_sci, 2])
            shifts_xy_sky = np.zeros([ndust, n_sky, self.new_ndit_sky, 2])
            crop_sz = int(3 * self.resel_ori)
            # to ensure crop size is odd. if its even, +1 to crop_sz
            if crop_sz % 2 == 0:
                crop_sz += 1
    
            # t0 = time_ini()
    
            # SCI frames
            bar = pyprind.ProgBar(n_sci, stream=1, title='Finding shifts to be applied to the SCI frames')
            for sc, fits_name in enumerate(sci_list):
                tmp_cube = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
                for zz in range(tmp_cube.shape[0]):
                    tmp = frame_filter_highpass(tmp_cube[zz], mode='median-subt', median_size=hpf_sz,
                                                kernel_size=hpf_sz, fwhm_size=self.fwhm)
                    for dd in range(ndust):
                        try:  # note we have to do try, because for some (rare) channels the gaussian fit fails
                            y_tmp, x_tmp = fit_2dgaussian(tmp, crop=True, cent=dust_xy[dd], cropsize=crop_sz,
                                                          fwhmx=self.resel_ori, fwhmy=self.resel_ori, full_output=False,
                                                          debug=False)
                        except ValueError:
                            x_tmp, y_tmp = dust_xy[dd]
                            if verbose:
                                print(
                                    "!!! Gaussian fit failed for sc #{}, dd #{}. We set position to first (eye-)guess position.".format(
                                        sc, dd), flush=True)
                        shifts_xy_sci[dd, sc, zz, 0] = xy_cube0[dd, 0] - x_tmp
                        shifts_xy_sci[dd, sc, zz, 1] = xy_cube0[dd, 1] - y_tmp
                bar.update()
    
            # SKY frames
            bar = pyprind.ProgBar(n_sky, stream=1, title='Finding shifts to be applied to the SKY frames')
            for sk, fits_name in enumerate(sky_list):
                tmp_cube = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
                for zz in range(tmp_cube.shape[0]):
                    tmp = frame_filter_highpass(tmp_cube[zz], mode='median-subt', median_size=hpf_sz,
                                                kernel_size=hpf_sz, fwhm_size=self.fwhm)
                    # check tmp after highpass filter
                    for dd in range(ndust):
                        try:
                            y_tmp, x_tmp = fit_2dgaussian(tmp, crop=True, cent=dust_xy[dd], cropsize=crop_sz,
                                                          fwhmx=self.resel_ori, fwhmy=self.resel_ori, full_output=False,
                                                          debug=False)
                        except ValueError:
                            x_tmp, y_tmp = dust_xy[dd]
                            if verbose:
                                print(
                                    "!!! Gaussian fit failed for sk #{}, dd #{}. We set position to first (eye-)guess position.".format(
                                        sc, dd), flush=True)
                        shifts_xy_sky[dd, sk, zz, 0] = xy_cube0[dd, 0] - x_tmp
                        shifts_xy_sky[dd, sk, zz, 1] = xy_cube0[dd, 1] - y_tmp
                bar.update()
            # time_fin(t0)
    
            # try to debug the fit, check dust pos
            if verbose:
                print("Max stddev of the shifts found for the {} dust grains: ".format(ndust),
                      np.amax(np.std(shifts_xy_sci, axis=0)))
                print("Min stddev of the shifts found for the {} dust grains: ".format(ndust),
                      np.amin(np.std(shifts_xy_sci, axis=0)))
                print("Median stddev of the shifts found for the {} dust grains: ".format(ndust),
                      np.median(np.std(shifts_xy_sci, axis=0)))
                print("Median shifts found for the {} dust grains (SCI): ".format(ndust),
                      np.median(np.median(np.median(shifts_xy_sci, axis=0), axis=0), axis=0))
                print("Median shifts found for the {} dust grains: (SKY)".format(ndust),
                      np.median(np.median(np.median(shifts_xy_sky, axis=0), axis=0), axis=0), flush=True)
    
            shifts_xy_sci_med = np.median(shifts_xy_sci, axis=0)
            shifts_xy_sky_med = np.median(shifts_xy_sky, axis=0)
    
            for sc, fits_name in enumerate(sci_list):
                try:
                    tmp = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
                    tmp_tmp_tmp_tmp = np.zeros_like(tmp)
                    for zz in range(tmp.shape[0]):
                        tmp_tmp_tmp_tmp[zz] = frame_shift(tmp[zz], shifts_xy_sci_med[sc, zz, 1],
                                                          shifts_xy_sci_med[sc, zz, 0], imlib=imlib)
                    write_fits(self.outpath + '3_AGPM_aligned_' + fits_name, tmp_tmp_tmp_tmp, verbose=debug)
                    if remove:
                        system("rm " + self.outpath + '3_rmfr_' + fits_name)
                except:
                    print("file #{} not found".format(sc), flush=True)
    
            for sk, fits_name in enumerate(sky_list):
                tmp = open_fits(self.outpath + '3_rmfr_' + fits_name, verbose=debug)
                tmp_tmp_tmp_tmp = np.zeros_like(tmp)
                for zz in range(tmp.shape[0]):
                    tmp_tmp_tmp_tmp[zz] = frame_shift(tmp[zz], shifts_xy_sky_med[sk, zz, 1], shifts_xy_sky_med[sk, zz, 0],
                                                      imlib=imlib)
                write_fits(self.outpath + '3_AGPM_aligned_' + fits_name, tmp_tmp_tmp_tmp, verbose=debug)
                if remove:
                    system("rm " + self.outpath + '3_rmfr_' + fits_name)
        else:
            for sc, fits_name in enumerate(sci_list):
                system('mv '+ self.outpath + '3_rmfr_' + fits_name + ' '+ self.outpath + '3_AGPM_aligned_' + fits_name)
            for sk, fits_name in enumerate(sky_list):
                system('mv '+ self.outpath + '3_rmfr_' + fits_name + ' '+ self.outpath + '3_AGPM_aligned_' + fits_name)

        # Finally, perform sky subtraction:
        ################## MEDIAN ##################################
        if mode == 'median':
            sci_list_test = [sci_list[0], sci_list[int(n_sci / 2)],
                             sci_list[-1]]  # first test then do with all sci_list

            master_skies2 = np.zeros([n_sky, self.final_sz, self.final_sz])
            master_sky_times = np.zeros(n_sky)

            for sk, fits_name in enumerate(sky_list):
                tmp_tmp_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + fits_name, verbose=debug)
                _, head_tmp = open_fits(self.inpath + fits_name, header=True, verbose=debug)
                master_skies2[sk] = np.median(tmp_tmp_tmp, axis=0)
                master_sky_times[sk] = head_tmp['MJD-OBS']
            write_fits(self.outpath + "master_skies.fits", master_skies2, verbose=debug)
            write_fits(self.outpath + "master_sky_times.fits", master_sky_times, verbose=debug)

            master_skies2 = open_fits(self.outpath + "master_skies.fits", verbose=debug)
            master_sky_times = open_fits(self.outpath + "master_sky_times.fits", verbose=debug)

            bar = pyprind.ProgBar(n_sci, stream=1, title='Subtracting sky with closest frame in time')
            for sc, fits_name in enumerate(sci_list_test):
                tmp_tmp_tmp_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + fits_name, verbose=debug)
                tmpSKY2 = np.zeros_like(tmp_tmp_tmp_tmp)  ###
                _, head_tmp = open_fits(self.inpath + fits_name, header=True, verbose=debug)
                sc_time = head_tmp['MJD-OBS']
                idx_sky = find_nearest(master_sky_times, sc_time)
                tmpSKY2 = tmp_tmp_tmp_tmp - master_skies2[idx_sky]
                write_fits(self.outpath + '4_sky_subtr_' + fits_name, tmpSKY2, verbose=debug)  ###
            bar.update()
            if plot:
                old_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[0])
                old_tmp_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[-1])
                tmp = open_fits(self.outpath + '4_sky_subtr_' + sci_list[0])
                tmp_tmp = open_fits(self.outpath + '4_sky_subtr_' + sci_list[-1])
                if old_tmp.ndim == 3:
                    old_tmp = np.median(old_tmp, axis=0)
                    old_tmp_tmp = np.median(old_tmp_tmp, axis=0)
                    tmp = np.median(tmp, axis=0)
                    tmp_tmp = np.median(tmp_tmp, axis=0)
                plot_frames((old_tmp, old_tmp_tmp, tmp, tmp_tmp), save=self.outpath + 'SCI_median_sky_subtraction')

        ############## PCA ##############

        if mode == 'PCA' or mode == 'pca':
            master_skies2 = np.zeros([n_sky, self.final_sz, self.final_sz])
            master_sky_times = np.zeros(n_sky)
            for sk, fits_name in enumerate(sky_list):
                tmp_tmp_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + fits_name, verbose=debug)
                _, head_tmp = open_fits(self.inpath + fits_name, header=True, verbose=debug)
                master_skies2[sk] = np.median(tmp_tmp_tmp, axis=0)
                master_sky_times[sk] = head_tmp['MJD-OBS']
            write_fits(self.outpath + "master_skies.fits", master_skies2, verbose=debug)
            write_fits(self.outpath + "master_sky_times.fits", master_sky_times, verbose=debug)

            all_skies = np.zeros([n_sky * self.new_ndit_sky, self.final_sz, self.final_sz])
            for sk, fits_name in enumerate(sky_list):
                tmp = open_fits(self.outpath + '3_AGPM_aligned_' + fits_name, verbose=debug)
                if tmp.ndim == 3:
                    all_skies[sk * self.new_ndit_sky:(sk + 1) * self.new_ndit_sky] = tmp[:self.new_ndit_sky]
                else:
                    all_skies[sk] = tmp.copy()
                    
            # Define mask for the region where the PCs will be optimal
            # make sure the mask avoids dark region.
            mask_arr = np.ones([self.com_sz, self.com_sz])
            mask_inner_rad = int(3 / self.dataset_dict['pixel_scale'])
            mask_width = int(self.shadow_r * 0.8 - mask_inner_rad)
            mask_AGPM = get_annulus_segments(mask_arr, mask_inner_rad, mask_width, mode='mask')[0]
            mask_AGPM = frame_crop(mask_AGPM, self.final_sz, verbose=debug)
            # Do PCA subtraction of the sky
            if plot:
                tmp_tmp = open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[-1], verbose=debug)
                if tmp.ndim == 3:
                    tmp = np.median(tmp, axis=0)
                    tmp_tmp = np.median(tmp_tmp, axis=0)
                plot_frames((tmp_tmp, tmp, mask_AGPM), vmin=(np.percentile(tmp_tmp, 0.1), np.percentile(tmp, 0.1), 0),
                            vmax=(np.percentile(tmp_tmp, 99.9), np.percentile(tmp, 99.9), 1),
                            label=('Science frame', 'Sky frame', 'Mask'), dpi=300,
                            save=self.outpath + 'PCA_sky_subtract_mask.pdf')

            if verbose:
                print('Beginning PCA subtraction', flush=True)

            if npc is None or isinstance(npc, list):  # checks whether none or list
                if npc is None:
                    nnpc_tmp = np.array([1, 2, 3, 4, 5, 10, 20, 40, 60])  # the number of principal components to test
                    # nnpc_tmp = np.array([1,2])
                else:
                    nnpc_tmp = npc  # takes the list
                nnpc = np.array([pc for pc in nnpc_tmp if pc < n_sky * self.new_ndit_sky])  # no idea

                ################### start new stuff

                test_idx = [0, int(len(sci_list) / 2),
                            len(sci_list) - 1]  # first, middle and last index in science list
                npc_opt = np.zeros(len(test_idx))  # array of zeros the length of the number of test cubes

                for sc, fits_idx in enumerate(test_idx):  # iterate over the 3 indices
                    _, head = open_fits(self.inpath + sci_list[fits_idx], verbose=debug,
                                        header=True)  # open the cube and get the header
                    sc_time = head['MJD-OBS']  # read this part of the header, float with the start time?
                    idx_sky = find_nearest(master_sky_times, sc_time)  # finds the corresponding cube using the time
                    tmp = open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[fits_idx],
                                    verbose=debug)  # opens science cube
                    if tmp.ndim == 2:
                        tmp = tmp[np.newaxis,:,:]
                    pca_lib = all_skies[int(np.sum(self.real_ndit_sky[:idx_sky])):int(
                        np.sum(self.real_ndit_sky[:idx_sky + 1]))]  # gets the sky cube?
                    med_sky = np.median(pca_lib, axis=0)  # takes median of the sky cubes
                    mean_std = np.zeros(
                        nnpc.shape[0])  # zeros array with length the number of principal components to test
                    hmean_std = np.zeros(nnpc.shape[0])  # same as above for some reason?
                    for nn, npc_tmp in enumerate(nnpc):  # iterate over the number of principal components to test
                        tmp_tmp = cube_subtract_sky_pca(tmp - med_sky, all_skies - med_sky,
                                                        mask_AGPM, ref_cube=None,
                                                        ncomp=npc_tmp)  # runs PCA sky subtraction
                        # write_fits(self.outpath+'4_sky_subtr_medclose1_npc{}_'.format(npc_tmp)+sci_list[fits_idx], tmp_tmp, verbose=debug)
                        # measure mean(std) in all apertures in tmp_tmp, and record for each npc
                        std = np.zeros(ndust_all)  # zeros array the length of the number of dust objects
                        for dd in range(ndust_all):  # iterate over the number of dust specks
                            std[dd] = np.std(get_circle(np.median(tmp_tmp, axis=0), 3 * self.fwhm, mode='val',
                                                        cy=dust_xy_all[dd][1], cx=dust_xy_all[dd][
                                    0]))  # standard deviation of the values in a circle around the dust in median sky cube??
                        mean_std[nn] = np.mean(std)  # mean of standard dev for that PC
                        std_sort = np.sort(std)  # sort std from smallest to largest?
                        hmean_std[nn] = np.mean(std_sort[
                                                int(ndust_all / 2.):])  # takes the mean of the higher std for second half of the dust specks?
                    npc_opt[sc] = nnpc[np.argmin(hmean_std)]  # index of the lowest standard deviation?
                    if verbose:
                        print("***** SCI #{:.0f} - OPTIMAL NPC = {:.0f} *****\n".format(sc, npc_opt[sc]), flush=True)
                npc = int(np.median(npc_opt))
                if verbose:
                    print('##### Optimal number of principal components for sky subtraction:', npc, '#####', flush=True)
                with open(self.outpath + "npc_sky_subtract.txt", "w") as f:
                    f.write('{}'.format(npc))
                write_fits(self.outpath + "TMP_npc_opt.fits", npc_opt, verbose=debug)
                ################ end new stuff

            #                bar = pyprind.ProgBar(n_sci, stream=1, title='Subtracting sky with PCA')
            #                for sc, fits_name in enumerate(sci_list):
            #                    _, head = open_fits(self.inpath+fits_name, verbose=debug, header=True)
            #                    sc_time = head['MJD-OBS']
            #                    idx_sky = find_nearest(master_sky_times,sc_time)
            #                    tmp = open_fits(self.outpath+'3_AGPM_aligned_'+fits_name, verbose=debug)
            #                    pca_lib = all_skies[int(np.sum(self.real_ndit_sky[:idx_sky])):int(np.sum(self.real_ndit_sky[:idx_sky+1]))]
            #                    med_sky = np.median(pca_lib,axis=0)
            #                    mean_std = np.zeros(nnpc.shape[0])
            #                    hmean_std = np.zeros(nnpc.shape[0])
            #                    for nn, npc_tmp in enumerate(nnpc):
            #                        tmp_tmp = cube_subtract_sky_pca(tmp-med_sky, all_skies-med_sky,
            #                                                                    mask_AGPM, ref_cube=None, ncomp=npc_tmp)
            #                        write_fits(self.outpath+'4_sky_subtr_medclose1_npc{}_'.format(npc_tmp)+fits_name, tmp_tmp, verbose=debug)
            #                        # measure mean(std) in all apertures in tmp_tmp, and record for each npc
            #                        std = np.zeros(ndust_all)
            #                        for dd in range(ndust_all):
            #                            std[dd] = np.std(get_circle(np.median(tmp_tmp,axis=0), 3*self.fwhm, mode = 'val',
            #                                                                cy=dust_xy_all[dd][1], cx=dust_xy_all[dd][0]))
            #                        mean_std[nn] = np.mean(std)
            #                        std_sort = np.sort(std)
            #                        hmean_std[nn] = np.mean(std_sort[int(ndust_all/2.):])
            #                    npc_opt[sc] = nnpc[np.argmin(hmean_std)]
            ##                    if verbose:
            ##                        print("***** SCI #{:.0f} - OPTIMAL NPC = {:.0f} *****\n".format(sc,npc_opt[sc]))
            #                    nnpc_bad = [pc for pc in nnpc if pc!=npc_opt[sc]]
            #                    if remove:
            #                        os.system("rm "+self.outpath+'3_AGPM_aligned_'+fits_name)
            #                        for npc_bad in nnpc_bad:
            #                            os.system("rm "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_bad)+fits_name)
            #                            os.system("mv "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_opt[sc])+fits_name + ' ' + self.outpath+'4_sky_subtr_'+fits_name)
            #                    else:
            #                        os.system("cp "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_opt[sc])+fits_name + ' ' + self.outpath+'4_sky_subtr_'+fits_name)

            #                    bar.update()

            #            if type(npc) is list:
            #                nnpc = np.array([pc for pc in npc if pc < n_sky*self.new_ndit_sky])
            #                npc_opt = np.zeros(len(sci_list))
            #                bar = pyprind.ProgBar(n_sci, stream=1, title='Subtracting sky with PCA')
            #                for sc, fits_name in enumerate(sci_list):
            #                    _, head = open_fits(self.inpath+fits_name, verbose=debug, header=True)
            #                    sc_time = head['MJD-OBS']
            #                    idx_sky = find_nearest(master_sky_times,sc_time)
            #                    tmp = open_fits(self.outpath+'3_AGPM_aligned_'+fits_name, verbose=debug)
            #                    pca_lib = all_skies[int(np.sum(self.real_ndit_sky[:idx_sky])):int(np.sum(self.real_ndit_sky[:idx_sky+1]))]
            #                    med_sky = np.median(pca_lib,axis=0)
            #                    mean_std = np.zeros(nnpc.shape[0])
            #                    hmean_std = np.zeros(nnpc.shape[0])
            #                    for nn, npc_tmp in enumerate(nnpc):
            #                        tmp_tmp = cube_subtract_sky_pca(tmp-med_sky, all_skies-med_sky,
            #                                                                    mask_AGPM, ref_cube=None, ncomp=npc_tmp)
            #                        write_fits(self.outpath+'4_sky_subtr_medclose1_npc{}_'.format(npc_tmp)+fits_name, tmp_tmp, verbose=debug) # this should be the most common output of the final calibrated cubes
            #                        # measure mean(std) in all apertures in tmp_tmp, and record for each npc
            #                        std = np.zeros(ndust_all)
            #                        for dd in range(ndust_all):
            #                            std[dd] = np.std(get_circle(np.median(tmp_tmp,axis=0), 3*self.fwhm, mode = 'val',
            #                                                                cy=dust_xy_all[dd][1], cx=dust_xy_all[dd][0]))
            #                        mean_std[nn] = np.mean(std)
            #                        std_sort = np.sort(std)
            #                        hmean_std[nn] = np.mean(std_sort[int(ndust_all/2.):])
            #                    npc_opt[sc] = nnpc[np.argmin(hmean_std)]
            #                    if verbose:
            #                        print("***** SCI #{:.0f} - OPTIMAL NPC = {:.0f} *****\n".format(sc,npc_opt[sc]))
            #                    nnpc_bad = [pc for pc in nnpc if pc!=npc_opt[sc]]
            #                    if remove:
            #                        os.system("rm "+self.outpath+'3_AGPM_aligned_'+fits_name)
            #                        os.system("mv "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_opt[sc])+fits_name + ' ' + self.outpath+'4_sky_subtr_'+fits_name)
            #                        for npc_bad in nnpc_bad:
            #                            os.system("rm "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_bad)+fits_name)
            #                    else:
            #                        os.system("cp "+self.outpath+'4_sky_subtr_medclose1_npc{:.0f}_'.format(npc_opt[sc])+fits_name + ' ' + self.outpath+'4_sky_subtr_'+fits_name)
            #                    bar.update()
            #                write_fits(self.outpath+"TMP_npc_opt.fits",npc_opt)

            # else: # goes into this loop after it has found the optimal number of pcs
            # bar = pyprind.ProgBar(n_sci, stream=1, title='Subtracting sky with PCA')
            for sc, fits_name in enumerate(sci_list):  # previously sci_list_test
                _, head = open_fits(self.inpath + sci_list[sc], verbose=debug,
                                    header=True)  # open the cube and get the header
                sc_time = head['MJD-OBS']  # read this part of the header, float with the start time?
                idx_sky = find_nearest(master_sky_times, sc_time)  # finds the corresponding cube using the time
                tmp = open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[sc],
                                verbose=debug)  # opens science cube
                if tmp.ndim == 2:
                    tmp = tmp[np.newaxis,:,:]
                pca_lib = all_skies[int(np.sum(self.real_ndit_sky[:idx_sky])):int(
                    np.sum(self.real_ndit_sky[:idx_sky + 1]))]  # gets the sky cube?
                med_sky = np.median(pca_lib, axis=0)  # takes median of the sky cubes
                tmp_tmp = cube_subtract_sky_pca(tmp - med_sky, all_skies - med_sky, 
                                                mask_AGPM, ref_cube=None,
                                                ncomp=npc)
                write_fits(self.outpath + '4_sky_subtr_' + fits_name, tmp_tmp, verbose=debug)
                # bar.update()
                if remove:
                    system("rm " + self.outpath + '3_AGPM_aligned_' + fits_name)

            if verbose:
                print('Finished PCA dark subtraction', flush=True)
            if plot:
                if npc is None:
                    # ... IF PCA WITH DIFFERENT NPCs
                    old_tmp = np.median(open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[-1], verbose=debug), axis=0)
                    tmp = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_'.format(1) + sci_list[-1], verbose=debug), axis=0)
                    tmp_tmp = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_'.format(5) + sci_list[-1], verbose=debug), axis=0)
                    tmp_tmp_tmp = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_'.format(100) + sci_list[-1], verbose=debug ), axis=0)
                    tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_no_shift_'.format(1) + sci_list[-1], verbose=debug), axis=0)
                    tmp_tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_no_shift_'.format(5) + sci_list[-1], verbose=debug), axis=0)
                    tmp_tmp_tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_npc{}_no_shift_'.format(100) + sci_list[-1]), verbose=debug, axis=0)

                    plot_frames((tmp, tmp_tmp, tmp_tmp_tmp, tmp2, tmp_tmp2, tmp_tmp_tmp2), dpi=300, rows=2,
                                cmap='inferno', save=self.outpath + 'SCI_PCA_sky_subtraction.pdf')
                else:
                    # ... IF PCA WITH A SPECIFIC NPC
                    old_tmp = np.median(open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[0], verbose=debug), axis=0)
                    old_tmp_tmp = np.median(open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[int(n_sci / 2)], verbose=debug), axis=0)
                    old_tmp_tmp_tmp = np.median(open_fits(self.outpath + '3_AGPM_aligned_' + sci_list[-1], verbose=debug), axis=0)
                    tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_' + sci_list[0], verbose=debug), axis=0)
                    tmp_tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_' + sci_list[int(n_sci / 2)], verbose=debug), axis=0)
                    tmp_tmp_tmp2 = np.median(open_fits(self.outpath + '4_sky_subtr_' + sci_list[-1], verbose=debug), axis=0)
                    plot_frames((old_tmp, old_tmp_tmp, old_tmp_tmp_tmp, tmp2, tmp_tmp2, tmp_tmp_tmp2), rows=2,
                                dpi=300, cmap='inferno', save=self.outpath + 'SCI_PCA_sky_subtraction.pdf')

        # time_fin(t0)

    def clean_fits(self):
        """
        Use this method to clean for any intermediate fits files
        """
        # be careful when using avoid removing PSF related fits
        # os.system("rm "+self.outpath+'common_sz.fits')
        # os.system("rm "+self.outpath+'real_ndit_sci_sky.fits')
        # os.system("rm "+self.outpath+'new_ndit_sci_sky_unsat.fits')
        # #os.system("rm "+self.outpath+'final_sz.fits')
        # os.system("rm "+self.outpath+'flat_dark_cube.fits')
        # os.system("rm "+self.outpath+'master_bpix_map.fits')
        # os.system("rm "+self.outpath+'master_bpix_map_2ndcrop.fits')
        # os.system("rm "+self.outpath+'master_bpix_map_unsat.fits')
        # os.system("rm "+self.outpath+'master_flat_field.fits')
        # os.system("rm "+self.outpath+'master_flat_field_unsat.fits')
        # os.system("rm "+self.outpath+'master_skies.fits')
        # os.system("rm "+self.outpath+'master_sky_times.fits')
        # #os.system("rm "+self.outpath+'master_unsat_psf.fits') these are needed in post processing
        # #os.system("rm "+self.outpath+'master_unsat_psf_norm.fits')
        # #os.system("rm "+self.outpath+'master_unsat-stellarpsf_fluxes.fits')
        # os.system("rm "+self.outpath+'shadow_median_frame.fits')
        # os.system("rm "+self.outpath+'sci_dark_cube.fits')
        # os.system("rm "+self.outpath+'sky_mjd_times.fits')
        # os.system("rm "+self.outpath+'TMP_2_master_median_SCI.fits')
        # os.system("rm "+self.outpath+'TMP_2_master_median_SKY.fits')
        # os.system("rm "+self.outpath+'TMP_med_bef_SKY_subtr.fits')
        # os.system("rm "+self.outpath+'TMP_npc_opt.fits')
        # os.system("rm "+self.outpath+'unsat_dark_cube.fits')
        system("rm " + self.outpath + '1_*.fits')
        system("rm " + self.outpath + '2_*.fits')
        system("rm " + self.outpath + '3_*.fits')
