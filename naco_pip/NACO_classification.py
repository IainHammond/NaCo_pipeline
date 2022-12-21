#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifies cubes and find derotation angles

@author: lewis, iain
"""
__author__ = 'Lewis Picker, Iain Hammond'
__all__ = ['input_dataset', 'find_AGPM']

import pdb
from os import listdir, makedirs
from os.path import isfile, join, isdir

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from photutils import CircularAperture, aperture_photometry

from hciplot import plot_frames
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_fix_badpix_isolated, approx_stellar_position
from vip_hci.var import frame_filter_lowpass, frame_center, get_square

matplotlib.use('Agg')


def find_AGPM(path, size=151, verbose=True, debug=False):
    """
    To prevent dust grains being picked up as the AGPM.

    This function will find the location of the AGPM/star (even when sky frames are mixed with science frames), by
    by creating a subset square image around the expected location and applies a low pass filter + max search method
    and returns the (y,x) location of the AGPM/star

    Parameters
    ----------
    path : str
        Path to cube
    size : int
        Pixel dimensions of the square to sample for the AGPM/star (ie size = 100 is 100 x 100 pixels)
    verbose : bool
        If True the final (x, y) location is printed.
    debug : bool, False by default
        Prints significantly more information.
    Returns
    ----------
    [ycom, xcom] : location of AGPM/star
    """
    cube = open_fits(path, verbose=debug)
    cy, cx = frame_center(cube, verbose=verbose)
    if cube.ndim == 3:
        median_frame = cube[-1]  # last frame is median
    else:
        median_frame = cube.copy()

    # define a square with the center being the approximate AGPM/star position
    median_frame, cornery, cornerx = get_square(median_frame, size=size, y=cy, x=cx, position=True, verbose=debug)
    # apply low pass filter
    median_frame = frame_filter_lowpass(median_frame, median_size=7, mode='median')
    median_frame = frame_filter_lowpass(median_frame, mode='gauss', fwhm_size=5)
    # find coordinates of max flux in the square
    ycom_tmp, xcom_tmp = np.unravel_index(np.argmax(median_frame), median_frame.shape)
    # AGPM/star is the bottom-left corner coordinates plus the location of the max in the square
    ycom = cornery + ycom_tmp
    xcom = cornerx + xcom_tmp

    if verbose:
        print('The (x,y) location of the AGPM/star is ({},{})'.format(xcom, ycom), flush=True)
    return [ycom, xcom]


class input_dataset():
    def __init__(self, inpath, outpath, dataset_dict, coro=True):
        self.inpath = inpath
        self.outpath = outpath
        self.coro = coro
        old_list = listdir(self.inpath)
        self.file_list = [file for file in old_list if file.endswith('.fits') and not file.startswith('M.')]
        self.dit_sci = dataset_dict['dit_sci']
        self.ndit_sci = dataset_dict['ndit_sci']
        self.ndit_sky = dataset_dict['ndit_sky']
        self.dit_unsat = dataset_dict['dit_unsat']
        self.ndit_unsat = dataset_dict['ndit_unsat']
        self.dit_flat = dataset_dict['dit_flat']
        self.dataset_dict = dataset_dict
        print('##### Number of raw FITS files:', len(self.file_list), '#####', flush=True)
        if not isdir(self.outpath):
            makedirs(self.outpath)

    def bad_columns(self, correct=True, overwrite=False, sat_val=32768, plot=True, verbose=True, debug=False):
        """
        In NACO data there are systematic bad columns in the lower left quadrant.
        This method will correct those bad columns with the median of the neighbouring pixels.
        May require manual inspection of one frame to confirm the saturated value.

        correct : bool, optional
            Whether to correct the bad column, which may not be present in old datasets.
            If False the files are simply moved to the outpath for the next pipeline step.
        overwrite : bool, optional
            Whether to re-run correction and overwrite the file if it already exists.
        sat_val : int, optional
            Value of the saturated column. Default 32768.
        plot : bool, optional
            Saves a before and after pdf of the first science frame/cube correction.
        verbose : bool, optional
            Prints starting and finishing messages.
        debug : bool, optional
            Full print output from each VIP function.
        """
        # creating bad pixel map
        # bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        # for i in range(3, 509, 8):
        #     for j in range(512):
        #         bcm[j,i] = 1

        for fname in self.file_list:
            if correct:
                if overwrite or not isfile(self.outpath + fname):
                    tmp, header_fname = open_fits(self.inpath + fname, header=True, verbose=debug)
                    # ADD code here that checks for bad column and updates the mask
                    if verbose:
                        print('Fixing {} of shape {} and type {}'.format(fname, tmp.shape,
                                                                         header_fname['HIERARCH ESO DPR TYPE']), flush=True)
                    # crop the bad pixel map to the same dimensions of the frames
                    if len(tmp.shape) == 3:
                        nz, _, _ = tmp.shape
                        tmp_median = np.median(tmp, axis=0)  # median frame of cube
                        bcm = np.zeros_like(tmp_median, dtype=np.int8)  # make mask the same dimensions as cube
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
                            tmp[j] = frame_fix_badpix_isolated(tmp[j], bpm_mask=bcm, correct_only=True, size=5,
                                                               protect_mask=False, verbose=debug)
                        write_fits(self.outpath + fname, tmp, header_fname, verbose=debug, output_verify='fix')
                    else:
                        print('File {} is not a cube ({})'.format(fname, header_fname['HIERARCH ESO DPR TYPE']), flush=True)
                        bcm = np.zeros_like(tmp, dtype=np.int8)  # make mask the same dimensions as frame
                        bcm[np.where(tmp == sat_val)] = 1
                        # for i in range(0,nx):
                        #     for j in range(0,ny):
                        #         if tmp[j,i] == sat_val:
                        #             bcm[j,i] = 1
                        # cy, cx = ny/2 , nx/2
                        # ini_y, fin_y = int(512-cy), int(512+cy)
                        # ini_x, fin_x = int(512-cx), int(512+cx)
                        # bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                        tmp = frame_fix_badpix_isolated(tmp, bpm_mask=bcm, correct_only=True, size=5,
                                                        protect_mask=False, verbose=debug)
                        write_fits(self.outpath + fname, tmp, header_fname, verbose=debug, output_verify='fix')
                    if verbose:
                        print('Fixed file {}'.format(fname), flush=True)

                    # plot the first science frame
                    if plot and header_fname['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                            header_fname['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                            header_fname['HIERARCH ESO DET DIT'] == self.dit_sci and \
                            header_fname['HIERARCH ESO DET NDIT'] in self.ndit_sci:
                        science_before = open_fits(self.inpath + fname, verbose=debug)
                        science_after = open_fits(self.outpath + fname, verbose=debug)
                        if len(science_before.shape) == 3:
                            science_before = np.median(science_before, axis=0)
                        if len(science_after.shape) == 3:
                            science_after = np.median(science_after, axis=0)
                        # note the cuts are the same on purpose for a fair comparison
                        plot_frames((science_before, bcm, science_after), label=('Before', 'Detected Pixels', 'After'),
                                    vmin=(np.percentile(science_after, 0.5), 0, np.percentile(science_after, 0.5)),
                                    vmax=(np.percentile(science_after, 99.5), 1, np.percentile(science_after, 99.5)),
                                    top_colorbar=True, cmap='inferno', horsp=0.2, label_pad=(5, 280), dpi=300,
                                    save=self.outpath + 'bad_columns_correction.pdf')
                        plot = False
            else:
                if overwrite or not isfile(self.outpath + fname):
                    tmp, header_fname = open_fits(self.inpath + fname, header=True, verbose=debug)
                    write_fits(self.outpath + fname, tmp, header_fname, verbose=debug, output_verify='fix')

    def mk_dico(self, coro=True, plot=True, verbose=True, debug=False):
        """
        Reads the header of each FITS file and sorts them based on DIT and NDIT provided in the run script.

        plot : bool, optional
            Produces a plot of the number of files and airmass vs. time in the flats. If no airmass is in the flat
            header it will use the median pixel value as a proxy.
        verbose : bool, optional
            Prints starting and finishing messages.
        debug : bool, optional
            Full print output from each VIP function, such as opening each file.
        """
        # creating a dictionary
        file_list = [f for f in listdir(self.outpath) if isfile(join(self.outpath, f))]
        fits_list = []
        sci_list = []
        sci_list_mjd = []
        sky_list = []
        sky_list_mjd = []
        unsat_list = []
        flat_list = []
        flat_list_mjd = []
        airmass = []
        flat_dark_list = []
        sci_dark_list = []
        unsat_dark_list = []

        if verbose:
            print('Creating dictionary', flush=True)
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

                elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                      header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                      header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                      header['HIERARCH ESO DET NDIT'] in self.ndit_sky) and \
                        cube.shape[0] > 0.8 * min(self.ndit_sky):  # avoid bad cubes

                    sky_list.append(fname)
                    sky_list_mjd.append(header['MJD-OBS'])

                elif header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                        header['HIERARCH ESO DET DIT'] == self.dit_unsat and \
                        header['HIERARCH ESO DET NDIT'] in self.ndit_unsat:
                    unsat_list.append(fname)

                elif 'FLAT,SKY' in header['HIERARCH ESO DPR TYPE']:
                    if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                        flat_list.append(fname)
                        flat_list_mjd.append(header['MJD-OBS'])
                        try:
                            airmass.append(str(header['AIRMASS']))
                        except:
                            pass

                elif 'DARK' in header['HIERARCH ESO DPR TYPE']:
                    if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                        flat_dark_list.append(fname)
                    if header['HIERARCH ESO DET DIT'] == self.dit_sci:
                        sci_dark_list.append(fname)
                    if header['HIERARCH ESO DET DIT'] == self.dit_unsat:
                        unsat_dark_list.append(fname)

        # if no appropriate flat-sky found, use flat-lamp
        if len(flat_list) == 0:
            for fname in file_list:
                if fname.endswith('.fits') and fname.startswith('NACO'): 
                    cube, header = open_fits(self.outpath + fname, 
                                             header=True, verbose=debug)
                    cond1 = 'FLAT,LAMP' in header['HIERARCH ESO DPR TYPE']
                    cond2 = header['HIERARCH ESO DET DIT'] == self.dit_flat
                    if cond1 & cond2:
                       flat_list.append(fname)
                       
        if len(flat_list) == 0:
            msg = "No appropriate flat fields found. Double-check requested FLAT DIT?"
            raise ValueError(msg)

        if len(airmass) > 0:
            no_airmass = False
            with open(self.outpath + "airmass.txt", "w") as f:
                for airmasses in airmass:
                    f.write(airmasses + '\n')
        else:
            print('WARNING: No airmass detected in flat header. Flats will be sorted by median pixel value later.', flush=True)
            no_airmass = True
            airmass = []
            for fl, flat_name in enumerate(flat_list):
                tmp = open_fits(self.outpath + flat_name, verbose=debug)
                airmass.append(np.median(tmp))

        if plot:
            plt.scatter(flat_list_mjd, airmass, label='Flat')
            plt.xlabel('Time [MJD]')
            plt.xticks(rotation=45)
            plt.ticklabel_format(useOffset=False, axis='x')  # to prevent scientific notation.
            plt.minorticks_on()
            plt.legend()
            plt.grid(alpha=0.1)
            if no_airmass:
                plt.ylabel('Median flat value')
                plt.savefig(self.outpath + 'Inferred_airmasses.pdf', bbox_inches='tight', pad_inches=0.1)
            else:
                plt.ylabel('Airmass')
                plt.savefig(self.outpath + 'Airmasses.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.close('all')

        with open(self.outpath + "sci_list_mjd.txt", "w") as f:
            for sci_time in sci_list_mjd:
                f.write(str(sci_time) + '\n')
        with open(self.outpath + "sky_list_mjd.txt", "w") as f:
            for sky_time in sky_list_mjd:
                f.write(str(sky_time) + '\n')
        with open(self.outpath + "sci_list.txt", "w") as f:
            for sci in sci_list:
                f.write(sci + '\n')
        with open(self.outpath + "sky_list.txt", "w") as f:
            for sky in sky_list:
                f.write(sky + '\n')
        with open(self.outpath + "unsat_list.txt", "w") as f:
            for unsat in unsat_list:
                f.write(unsat + '\n')
        with open(self.outpath + "unsat_dark_list.txt", "w") as f:
            for unsat_dark in unsat_dark_list:
                f.write(unsat_dark + '\n')
        with open(self.outpath + "flat_dark_list.txt", "w") as f:
            for flat_dark in flat_dark_list:
                f.write(flat_dark + '\n')
        with open(self.outpath + "sci_dark_list.txt", "w") as f:
            for sci_dark in sci_dark_list:
                f.write(sci_dark + '\n')
        with open(self.outpath + "flat_list.txt", "w") as f:
            for flat in flat_list:
                f.write(flat + '\n')

        if verbose:
            print('Done :)', flush=True)

    def find_sky_in_sci_cube(self, nres=3, coro=True, plot=True, verbose=True, debug=False):
        """
        Empty SKY list could be caused by a misclassification of the header in NACO data.
        This method will check the flux of the SCI cubes around the location of the AGPM.
        A SKY cube should be less bright at that location allowing the separation of cubes.

        nres : float, optional
            Number of resolution elements
        coro : bool, optional
            If an AGPM was used or not during the observation.
        plot : bool, optional
            Save plot of the flux measured in all science frames and saves a bar graph of the final breakdown of
            each file type.
        verbose : bool, optional
            Prints information about the location of the star and flux.
        debug : bool, optional
            Significantly more information printed.
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

        test_cube = open_fits(self.outpath + sci_list[0], verbose=debug)
        if coro:
            yx = find_AGPM(self.outpath + sci_list[0], verbose=debug, debug=debug)
        elif not coro:
            yx = approx_stellar_position(test_cube, fwhm=self.resel + 1, verbose=debug)
            yx = (np.median(yx[:, 1]), np.median(yx[:, 0]))

        if plot:
            plot_frames(test_cube[-1], vmin=np.percentile(test_cube[-1], 0.5), vmax=np.percentile(test_cube[-1], 99.5),
                        cmap='inferno', dpi=300, circle_label='Inferred star/AGPM position', circle=(int(yx[1]), int(yx[0])),
                        circle_radius=3*(self.resel+1), circle_alpha=1, label_size=8, label=sci_list[0],
                        save=self.outpath + 'Inferred_star-AGPM_position.pdf')

        if verbose:
            print('The rough location of the star/AGPM is', 'y=', yx[0], 'x=', yx[1], flush=True)
            print('Measuring flux in SCI cubes...', flush=True)

        # create the aperture
        circ_aper = CircularAperture((yx[1], yx[0]), round(nres * self.resel))

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
                print('centre flux has been measured for', fname, flush=True)

        median_flux = np.median(flux_list)
        sd_flux = np.std(flux_list)

        if verbose:
            print('Sorting Sky from Sci', flush=True)

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
            plt.ylabel('Normalised flux')
            plt.xlabel('Cube')
            plt.savefig(self.outpath + 'flux_plot.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.close('all')

        sci_list.sort()
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

        if plot:
            unsat_list = []
            flat_list = []
            sci_dark_list = []
            unsat_dark_list = []
            flat_dark_list = []
            with open(self.outpath + "unsat_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    unsat_list.append(line.split('\n')[0])
            with open(self.outpath + "flat_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    flat_list.append(line.split('\n')[0])
            with open(self.outpath + "sci_dark_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    sci_dark_list.append(line.split('\n')[0])
            with open(self.outpath + "unsat_dark_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    unsat_dark_list.append(line.split('\n')[0])
            with open(self.outpath + "flat_dark_list.txt", "r") as f:
                tmp = f.readlines()
                for line in tmp:
                    flat_dark_list.append(line.split('\n')[0])

            types = ['OBJ', 'SKY', 'UNSAT', 'FLAT', 'SCI DARK', 'UNSAT DARK', 'FLAT DARK']
            count = [len(sci_list), len(sky_list), len(unsat_list), len(flat_list), len(sci_dark_list),
                     len(unsat_dark_list), len(flat_dark_list)]
            plt.bar(x=types, height=count, zorder=2, color=['black', 'red', 'green', 'blue', 'cyan', 'grey', 'purple'])
            plt.ylabel('Number of files')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.1, zorder=1)
            plt.savefig(self.outpath + 'File_summary.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.close('all')

        if len(sci_list_mjd) != len(sci_list):
            print('======== WARNING: SCI observation time list is a different length to SCI cube list!! ========', flush=True)

        if len(sky_list_mjd) != len(sky_list):
            print('======== WARNING: SKY observation time list is a different length to SKY cube list!! ========', flush=True)
        if verbose:
            print('done sorting :)', flush=True)

    def find_derot_angles(self, plot=True, verbose=True, debug=False):
        """ 
        For datasets with significant rotation when the telescope derotator is switched off. Finds the derotation
        angle vector to apply to a set of NACO cubes to align it with North up. Derotated offset is included here.
        Requires previous classification steps must have been completed.

        plot : bool, optional
            Saves a plot of the derotation angle vs time as a pdf.
        verbose: bool, optional
            Prints important information regarding the sequence.
        debug : bool, optional
            Whether to print each angle as it is computed.
            
        Writes to FITS file:
        --------------------
        derot_angles_uncropped: 2d numpy array (n_cubes x n_frames_max)
            vector of n_frames derot angles for each cube
            Important: n_frames may be different from one cube to the other!
            For cubes where n_frames < n_frames_max the last values of the row are padded with zeros.
        """
        # open the list of science images and add them to sci_list to be used in _derot_ang_ipag
        sci_list = []
        with open(self.outpath + "sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])
        sci_list.sort()

        print('Calculating derotation angles from header data...', flush=True)

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
                cube, header = open_fits(self.inpath + sci_list[ff], header=True, verbose=debug)
                n_frames_vec[ff] = cube.shape[0] - 1  # "-1" is because the last frame is the median of all others
                parang[ff] = header[kw_par]
                posang[ff] = header[kw_pos]
                pupilpos = 180.0 - parang[ff] + posang[ff]
                rot_pt_off[ff] = 90 + 89.44 - pupilpos
                if debug:
                    print("parang: {}, posang: {}, rot_pt_off: {}".format(parang[ff], posang[ff], rot_pt_off[ff]), flush=True)

            # NEXT CHECK IF THE OBSERVATION WENT THROUGH TRANSIT (change of sign in parang OR stddev of rot_pt_off > 1.)       

            rot_pt_off_med = np.median(rot_pt_off)
            rot_pt_off_std = np.std(rot_pt_off)

            if np.min(parang) * np.max(parang) < 0. or rot_pt_off_std > 1.:
                if verbose:
                    print("The observation goes through transit and/or the pupil position was reset in the middle of the observation: ", flush=True)
                    if np.min(parang) * np.max(parang) < 0.:
                        print("min/max parang: ", np.min(parang), np.max(parang), flush=True)
                    if rot_pt_off_std > 1.:
                        print("the standard deviation of pupil positions is greater than 1: ", rot_pt_off_std, flush=True)
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
                            n_changes), flush=True)
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

        final_derot_angs = np.zeros([n_sci, int(np.amax(n_frames_vec))])
        for sc in range(n_sci):
            n_frames = int(n_frames_vec[sc])
            nfr_vec = np.arange(n_frames)
            final_derot_angs[sc, :n_frames] = derot_angles_st[sc] + (
                        (derot_angles_nd[sc] - derot_angles_st[sc]) * nfr_vec / (n_frames - 1))

        write_fits(self.outpath + "derot_angles_uncropped.fits", final_derot_angs, verbose=debug)
        if verbose:
            print('Derotation angles have been computed and saved to file', flush=True)

        if plot:
            plt.plot(final_derot_angs[:,0])  # plot the first one from each cube
            plt.xlabel('Science cube')
            plt.ylabel('Starting derotation angle [deg]')
            plt.minorticks_on()
            plt.grid(alpha=0.1)
            plt.savefig(self.outpath + 'Derotation_angle_vector.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.close('all')
