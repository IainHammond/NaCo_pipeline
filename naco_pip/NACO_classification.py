#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:04 2020

@author: lewis, iain
"""
__author__ = 'Lewis Picker, Iain Hammond'
__all__ = ['input_dataset', 'find_AGPM_or_star']

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from photutils import CircularAperture, aperture_photometry
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_fix_badpix_isolated
from vip_hci.var import frame_filter_lowpass, frame_center, get_square
import pdb


def find_AGPM_or_star(self, file_list, rel_AGPM_pos_xy=(50.5, 6.5), size=101, verbose=True, debug=False):
    """
        added by Iain to prevent dust grains being picked up as the AGPM
        
        This method will find the location of the AGPM or star (even when sky frames are mixed with science frames), by
        using the known relative distance of the AGPM from the frame center in all VLT/NaCO datasets. It then creates a
        subset square image around the expected location and applies a low pass filter + max search method and returns
        the (y,x) location of the AGPM/star
        
        Parameters
        ----------
        file_list : list of str
            List containing all science cube names
        rel_AGPM_pos_xy : tuple, float
            relative location of the AGPM from the frame center in pixels, should be left unchanged. This is used to
            calculate how many pixels in x and y the AGPM is from the center and can be applied to almost all datasets
            with VLT/NaCO as the AGPM is always in the same approximate position
        size : int
            pixel dimensions of the square to sample for the AGPM/star (ie size = 100 is 100 x 100 pixels)
        verbose : bool
            If True extra messages are shown
        debug : bool, False by default
            Enters pdb once the location has been found

        Returns
        ----------
        [ycom, xcom] : location of AGPM or star        
        """
    sci_cube = open_fits(self.inpath + file_list[0])  # opens first sci/sky cube
    nz, ny, nx = sci_cube.shape  # gets size of it. science and sky cubes have same shape. assumes all cubes are the same ny and nx (they should be!)

    cy, cx = frame_center(sci_cube, verbose=verbose)  # find central pixel coordinates
    # then the position will be that plus the relative shift in y and x
    rel_shift_x = rel_AGPM_pos_xy[
        0]  # 50.5 is pixels from frame center to AGPM in x in an example data set, thus providing the relative shift
    rel_shift_y = rel_AGPM_pos_xy[
        1]  # 6.5 is pixels from frame center to AGPM in y in an example data set, thus providing the relative shift

    # the center of the square to apply the low pass filter to - is the approximate position of the AGPM/star based on previous observations
    y_tmp = cy + rel_shift_y
    x_tmp = cx + rel_shift_x
    median_all_cubes = np.zeros([len(file_list), ny, nx])  # makes empty array
    for sc, fits_name in enumerate(file_list):  # loops over all images
        tmp = open_fits(self.inpath + fits_name, verbose=debug)  # opens the cube
        median_all_cubes[sc] = tmp[
            -1]  # takes the last entry (the median) and adds it to the empty array
    median_frame = np.median(median_all_cubes, axis=0)  # median of all median frames

    # define a square of 100 x 100 with the center being the approximate AGPM/star position
    median_frame, cornery, cornerx = get_square(median_frame, size=size, y=y_tmp, x=x_tmp, position=True, verbose=True)
    # apply low pass filter
    median_frame = frame_filter_lowpass(median_frame, median_size=7, mode='median')
    median_frame = frame_filter_lowpass(median_frame, mode='gauss', fwhm_size=5)
    # find coordiates of max flux in the square
    ycom_tmp, xcom_tmp = np.unravel_index(np.argmax(median_frame), median_frame.shape)
    # AGPM/star is the bottom-left corner coordinates plus the location of the max in the square
    ycom = cornery + ycom_tmp
    xcom = cornerx + xcom_tmp

    if verbose:
        print('The location of the AGPM/star is', 'ycom =', ycom, 'xcom =', xcom)
    if debug:
        pdb.set_trace()
    return [ycom, xcom]


class input_dataset():
    def __init__(self, inpath, outpath, dataset_dict, coro=True):
        self.inpath = inpath
        self.outpath = outpath
        self.coro = coro
        old_list = os.listdir(self.inpath)
        self.file_list = [file for file in old_list if file.endswith('.fits')]
        self.dit_sci = dataset_dict['dit_sci']
        self.ndit_sci = dataset_dict['ndit_sci']
        self.ndit_sky = dataset_dict['ndit_sky']
        self.dit_unsat = dataset_dict['dit_unsat']
        self.ndit_unsat = dataset_dict['ndit_unsat']
        self.dit_flat = dataset_dict['dit_flat']
        self.fast_reduction = dataset_dict['fast_reduction']
        self.dataset_dict = dataset_dict
        print('##### Number of fits files:', len(self.file_list), '#####')

    def bad_columns(self, sat_val=32768, verbose=True, debug=False):
        """
        In NACO data there are systematic bad columns in the lower left quadrant
        This method will correct those bad columns with the median of the neighbouring pixels.
        May require manual inspection of one frame to confirm the saturated value.

        sat_val : int, optional
            value of the saturated column. Default 32768
        """
        # creating bad pixel map
        # bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        # for i in range(3, 509, 8):
        #     for j in range(512):
        #         bcm[j,i] = 1

        for fname in self.file_list:
            tmp, header_fname = open_fits(self.inpath + fname,
                                          header=True, verbose=debug)
            print('Opened {} of type {}'.format(fname, header_fname['HIERARCH ESO DPR TYPE']))
            # ADD code here that checks for bad column and updates the mask
            if verbose:
                print('Fixing {} of shape {}'.format(fname, tmp.shape))
            # crop the bad pixel map to the same dimensions of the frames

            if len(tmp.shape) == 3:
                nz, ny, nx = tmp.shape
                bcm = np.zeros((ny, nx), dtype=np.int8)  # make mask the same dimensions as cube
                tmp_median = np.median(tmp, axis=0)  # median frame of cube
                # loop through the median cube pixels and if any are 32768, add the location to the mask
                bcm[np.where(tmp_median == sat_val)] = 1
                # for i in range(0,nx): # all x pixels
                #     for j in range(0,ny): # all y pixels
                #         if tmp_median[j,i] == sat_val: # if saturated
                #             bcm[j,i] = 1 # mark as bad in mask
                # cy, cx = ny/2 , nx/2
                # ini_y, fin_y = int(512-cy), int(512+cy)
                # ini_x, fin_x = int(512-cx), int(512+cx)
                # bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                for j in range(nz):
                    # replace bad columns in each frame of the cubes
                    tmp[j] = frame_fix_badpix_isolated(tmp[j],
                                                       bpm_mask=bcm, sigma_clip=3,
                                                       num_neig=5, size=5, protect_mask=False,
                                                       radius=30, verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify='fix')

            else:
                print('File {} is not a cube ({})'.format(fname, header_fname['HIERARCH ESO DPR TYPE']))
                ny, nx = tmp.shape
                bcm = np.zeros((ny, nx), dtype=np.int8)  # make mask the same dimensions as frame
                bcm[np.where(tmp == sat_val)] = 1
                # for i in range(0,nx):
                #     for j in range(0,ny):
                #         if tmp[j,i] == sat_val:
                #             bcm[j,i] = 1
                # cy, cx = ny/2 , nx/2
                # ini_y, fin_y = int(512-cy), int(512+cy)
                # ini_x, fin_x = int(512-cx), int(512+cx)
                # bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                tmp = frame_fix_badpix_isolated(tmp,
                                                bpm_mask=bcm, sigma_clip=3, num_neig=5,
                                                size=5, protect_mask=False, radius=30,
                                                verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify='fix')
            if verbose:
                print('Fixed', fname)

    def mk_dico(self, coro=True, verbose=True, debug=False):
        if coro:
            # creating a dictionary
            file_list = [f for f in listdir(self.outpath) if
                         isfile(join(self.outpath, f))]
            fits_list = []
            sci_list = []
            sci_list_mjd = []
            sky_list = []
            sky_list_mjd = []
            unsat_list = []
            flat_list = []
            flat_dark_list = []
            sci_dark_list = []
            unsat_dark_list = []
            master_mjd = []
            master_airmass = []

            if verbose:
                print('Creating dictionary')
            for fname in file_list:
                if fname.endswith('.fits') and fname.startswith('NACO'):
                    fits_list.append(fname)
                    cube, header = open_fits(self.outpath + fname, header=True,
                                             verbose=debug)
                    if header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                            header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                            header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                            header['HIERARCH ESO DET NDIT'] in self.ndit_sci and \
                            cube.shape[0] > 0.8 * min(self.ndit_sci):  # avoid bad cubes

                        sci_list.append(fname)
                        sci_list_mjd.append(header['MJD-OBS'])
                        # sci_list_airmass.append(header['AIRMASS'])

                    elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                          header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                          header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                          header['HIERARCH ESO DET NDIT'] in self.ndit_sky) and \
                            cube.shape[0] > 0.8 * min(self.ndit_sky):  # avoid bad cubes

                        sky_list.append(fname)
                        sky_list_mjd.append(header['MJD-OBS'])
                        # sky_list_airmass.append(header['AIRMASS'])

                    elif header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                            header['HIERARCH ESO DET DIT'] == self.dit_unsat and \
                            header['HIERARCH ESO DET NDIT'] in self.ndit_unsat:
                        unsat_list.append(fname)
                        # unsat_list_mjd.append(header['MJD-OBS'])
                        # unsat_list_airmass.append(header['AIRMASS'])

                    elif 'FLAT,SKY' in header['HIERARCH ESO DPR TYPE']:
                        flat_list.append(fname)
                        # flat_list_mjd.append(header['MJD-OBS'])
                        # flat_list_airmass.append(header['AIRMASS'])

                    elif 'DARK' in header['HIERARCH ESO DPR TYPE']:
                        if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                            flat_dark_list.append(fname)
                        if header['HIERARCH ESO DET DIT'] == self.dit_sci:
                            sci_dark_list.append(fname)
                        if header['HIERARCH ESO DET DIT'] == self.dit_unsat:
                            unsat_dark_list.append(fname)

            with open(self.outpath + "sci_list_mjd.txt", "w") as f:
                for time in sci_list_mjd:
                    f.write(str(time) + '\n')
            with open(self.outpath + "sky_list_mjd.txt", "w") as f:
                for time in sky_list_mjd:
                    f.write(str(time) + '\n')
            with open(self.outpath + "sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci + '\n')
            with open(self.outpath + "sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci + '\n')
            with open(self.outpath + "unsat_list.txt", "w") as f:
                for sci in unsat_list:
                    f.write(sci + '\n')
            with open(self.outpath + "unsat_dark_list.txt", "w") as f:
                for sci in unsat_dark_list:
                    f.write(sci + '\n')
            with open(self.outpath + "flat_dark_list.txt", "w") as f:
                for sci in flat_dark_list:
                    f.write(sci + '\n')
            with open(self.outpath + "sci_dark_list.txt", "w") as f:
                for sci in sci_dark_list:
                    f.write(sci + '\n')
            with open(self.outpath + "flat_list.txt", "w") as f:
                for sci in flat_list:
                    f.write(sci + '\n')
            if verbose:
                print('Done :)')

    def find_sky_in_sci_cube(self, nres=3, coro=True, verbose=True, plot=None, debug=False):
        """
       Empty SKY list could be caused by a misclassification of the header in NACO data
       This method will check the flux of the SCI cubes around the location of the AGPM 
       A SKY cube should be less bright at that location allowing the separation of cubes
       
       """

        flux_list = []
        fname_list = []
        sci_list = []
        sky_list = []
        sci_list_mjd = []  # observation time of each sci cube
        sky_list_mjd = []  # observation time of each sky cube
        with open(self.outpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

        with open(self.outpath + "sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

        with open(self.outpath + "sci_list_mjd.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list_mjd.append(float(line.split('\n')[0]))

        with open(self.outpath + "sky_list_mjd.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list_mjd.append(float(line.split('\n')[0]))

        self.resel = (self.dataset_dict['wavelength'] * 180 * 3600) / (self.dataset_dict['size_telescope'] * np.pi *
                                                                       self.dataset_dict['pixel_scale'])

        agpm_pos = find_AGPM_or_star(self, sci_list, verbose=debug)
        if verbose:
            print('The rough location of the star/AGPM is', 'y  = ', agpm_pos[0], 'x =', agpm_pos[1])
            print('Measuring flux in SCI cubes...')

        # create the aperture
        circ_aper = CircularAperture((agpm_pos[1], agpm_pos[0]), round(nres * self.resel))

        # total flux through the aperture
        for fname in sci_list:
            cube_fname = open_fits(self.outpath + fname, verbose=debug)
            median_frame = np.median(cube_fname, axis=0)
            circ_aper_phot = aperture_photometry(median_frame,
                                                 circ_aper, method='exact')
            # append it to the flux list.
            circ_flux = np.array(circ_aper_phot['aperture_sum'])
            flux_list.append(circ_flux[0])
            fname_list.append(fname)
            if debug:
                print('centre flux has been measured for', fname)

        median_flux = np.median(flux_list)
        sd_flux = np.std(flux_list)

        if verbose:
            print('Sorting Sky from Sci')

        for i in range(len(flux_list)):
            if flux_list[i] < median_flux - 2 * sd_flux:
                sky_list.append(fname_list[i])  # add the sky cube to the sky cube list
                sky_list_mjd.append(sci_list_mjd[i])  # add the observation to the sky obs list from the sci obs list

                sci_list.remove(fname_list[i])  # remove the sky cube from the sci list
                sci_list_mjd.remove(sci_list_mjd[i])  # remove the sky obs time from the sci obs list
                symbol = 'bo'
            if plot:
                if flux_list[i] > median_flux - 2 * sd_flux:
                    symbol = 'go'
                else:
                    symbol = 'ro'
                plt.plot(i, flux_list[i] / median_flux, symbol)
        if plot:
            plt.title('Normalised flux around star')
            plt.ylabel('normalised flux')
            plt.xlabel('cube')
            if plot == 'save':
                plt.savefig(self.outpath + 'flux_plot.pdf')
            if plot == 'show':
                plt.show()

        # sci_list.sort()
        # with open(self.outpath + "sci_list.txt", "w") as f:
        #     for sci in sci_list:
        #         f.write(sci + '\n')
        sci_list.sort()
        if self.fast_reduction:
            with open(self.outpath + "sci_list_ori.txt", "w") as f:
                for ss, sci in enumerate(sci_list):
                    tmp = open_fits(self.inpath + sci, verbose=debug)
                    if ss == 0:
                        master_sci = np.zeros([len(sci_list), tmp.shape[1], tmp.shape[2]])
                    master_sci[ss] = tmp[-1]
                    f.write(sci + '\n')
            with open(self.outpath + "sci_list.txt", "w") as f:
                f.write('master_sci_fast_reduction.fits')
            write_fits(self.outpath + 'master_sci_fast_reduction.fits', master_sci, verbose=debug)
            print('Saved fast reduction master science cube')
        else:
            with open(self.outpath + "sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci + '\n')
        sky_list.sort()
        with open(self.outpath + "sky_list.txt", "w") as f:
            for sky in sky_list:
                f.write(sky + '\n')
        sci_list_mjd.sort()
        # save the sci observation time to text file
        with open(self.outpath + "sci_list_mjd.txt", "w") as f:
            for time in sci_list_mjd:
                f.write(str(time) + '\n')

        sky_list_mjd.sort()
        # save the sky observation time to text file
        with open(self.outpath + "sky_list_mjd.txt", "w") as f:
            for time in sky_list_mjd:
                f.write(str(time) + '\n')

        if len(sci_list_mjd) != len(sci_list):
            print('======== WARNING: SCI observation time list is a different length to SCI cube list!! ========')

        if len(sky_list_mjd) != len(sky_list):
            print('======== WARNING: SKY observation time list is a different length to SKY cube list!! ========')
        if verbose:
            print('done :)')

    ####### Iain's addition to find the derotation angles of the data ########

    def find_derot_angles(self, verbose=False):
        """ 
        For datasets with significant rotation when the telescope derotator is switched off.
        Requires sci_list.txt to exist in the outpath, thus previous classification steps must have been completed.
        Finds the derotation angle vector to apply to a set of NACO cubes to align it with North up. 
        IMPORTANT: The list of fits should be in chronological order of acquisition, however the code should sort them itself.
        
        verbose: str
            Whether to print the derotation angles as they are computed
            
        Returns:
        ********
        derot_angles: 2d numpy array (n_cubes x n_frames_max)
            vector of n_frames derot angles for each cube
            Important: n_frames may be different from one cube to the other!
            For cubes where n_frames < n_frames_max the last values of the row are padded with zeros.
        n_frames_vec: 1d numpy array
            Vector with number of frames in each cube
        """
        # open the list of science images and add them to sci_list to be used in _derot_ang_ipag
        sci_list = []
        if self.fast_reduction:
            with open(self.outpath + "sci_list_ori.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    sci_list.append(line.split('\n')[0])
        else:
            with open(self.outpath + "sci_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    sci_list.append(line.split('\n')[0])
        sci_list.sort()

        print('Calculating derotation angles from header data...')

        def _derot_ang_ipag(self, sci_list=sci_list, loc='st'):
            nsci = len(sci_list)
            parang = np.zeros(nsci)
            posang = np.zeros(nsci)
            rot_pt_off = np.zeros(nsci)
            n_frames_vec = np.ones(nsci, dtype=int)

            if loc == 'st':
                kw_par = 'HIERARCH ESO TEL PARANG START'  # Parallactic angle at start
                kw_pos = 'HIERARCH ESO ADA POSANG'  # Position angle at start
            elif loc == 'nd':
                kw_par = 'HIERARCH ESO TEL PARANG END'  # Parallactic angle at end
                kw_pos = 'HIERARCH ESO ADA POSANG END'  # Position angle at exposure end
            # FIRST COMPILE PARANG, POSANG and PUPILPOS 
            for ff in range(len(sci_list)):
                cube, header = open_fits(self.inpath + sci_list[ff], header=True, verbose=False)
                n_frames_vec[ff] = cube.shape[0] - 1  # "-1" is because the last frame is the median of all others
                parang[ff] = header[kw_par]
                posang[ff] = header[kw_pos]
                pupilpos = 180.0 - parang[ff] + posang[ff]
                rot_pt_off[ff] = 90 + 89.44 - pupilpos
                if verbose:
                    print("parang: {}, posang: {}, rot_pt_off: {}".format(parang[ff], posang[ff], rot_pt_off[ff]))

            # NEXT CHECK IF THE OBSERVATION WENT THROUGH TRANSIT (change of sign in parang OR stddev of rot_pt_off > 1.)       

            rot_pt_off_med = np.median(rot_pt_off)
            rot_pt_off_std = np.std(rot_pt_off)

            if np.min(parang) * np.max(parang) < 0. or rot_pt_off_std > 1.:
                if verbose:
                    print(
                        "The observation goes through transit and/or the pupil position was reset in the middle of the observation: ")
                    if np.min(parang) * np.max(parang) < 0.:
                        print("min/max parang: ", np.min(parang), np.max(parang))
                    if rot_pt_off_std > 1.:
                        print("the standard deviation of pupil positions is greater than 1: ", rot_pt_off_std)
                # find index where the transit occurs (change of sign of parang OR big difference in pupil pos)
                n_changes = 0
                for ff in range(len(sci_list) - 1):
                    if parang[ff] * parang[ff + 1] < 0. or np.abs(rot_pt_off[ff] - rot_pt_off[ff + 1]) > 1.:
                        idx_transit = ff + 1
                        n_changes += 1
                # check that these conditions only detected one passage through transit
                if n_changes != 1:
                    print(
                        " {} passages of transit were detected (instead of 1!). Check that the input fits list is given in chronological order.".format(
                            n_changes))
                    pdb.set_trace()

                rot_pt_off_med1 = np.median(rot_pt_off[:idx_transit])
                rot_pt_off_med2 = np.median(rot_pt_off[idx_transit:])

                final_derot_angs = rot_pt_off_med1 - parang
                final_derot_angs[idx_transit:] = rot_pt_off_med2 - parang[idx_transit:]

            else:
                final_derot_angs = rot_pt_off_med - parang

            # MAKE SURE ANGLES ARE IN THE RANGE (-180,180)deg
            min_derot_angs = np.amin(final_derot_angs)
            nrot_min = min_derot_angs / 360.
            if nrot_min < -0.5:
                final_derot_angs[np.where(final_derot_angs < -180)] = final_derot_angs[
                                                                          np.where(final_derot_angs < -180)] + np.ceil(
                    nrot_min) * 360.
            max_derot_angs = np.amax(final_derot_angs)
            nrot_max = max_derot_angs / 360.
            if nrot_max > 0.5:
                final_derot_angs[np.where(final_derot_angs > 180)] = final_derot_angs[
                                                                         np.where(final_derot_angs > 180)] - np.ceil(
                    nrot_max) * 360.

            return -1. * final_derot_angs, n_frames_vec

        n_sci = len(sci_list)
        derot_angles_st, _ = _derot_ang_ipag(self, sci_list, loc='st')
        derot_angles_nd, n_frames_vec = _derot_ang_ipag(self, sci_list, loc='nd')

        # if self.fast_reduction:
        # final_derot_angs = np.zeros([1, n_sci])

        final_derot_angs = np.zeros([n_sci, int(np.amax(n_frames_vec))])
        for sc in range(n_sci):
            n_frames = int(n_frames_vec[sc])
            nfr_vec = np.arange(n_frames)
            final_derot_angs[sc, :n_frames] = derot_angles_st[sc] + (
                        (derot_angles_nd[sc] - derot_angles_st[sc]) * nfr_vec / (n_frames - 1))

        if self.fast_reduction:
            final_derot_angs_median = np.zeros([n_sci])
            for sc in range(n_sci):
                # final_derot_angs[sc] = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, final_derot_angs[sc])
                final_derot_angs_median[sc] = np.median(final_derot_angs[sc])
            final_derot_angs = final_derot_angs_median
        write_fits(self.outpath + "derot_angles_uncropped.fits", final_derot_angs, verbose=verbose)
        print('Derotation angles have been computed and saved to file')
        # return final_derot_angs, n_frames_vec
