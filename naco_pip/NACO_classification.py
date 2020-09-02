#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:04 2020

@author: lewis
"""
__author__ = 'Lewis Picker'
__all__ = ['input_dataset','find_AGPM_list']
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
from vip_hci.var import frame_filter_lowpass
from naco_pip import fits_info
import pdb

#test = input_dataset('/home/lewis/Documents/Exoplanets/data_sets/HD179218/Tests/','/home/lewis/Documents/Exoplanets/data_sets/HD179218/Debug/')


def find_AGPM_list(self, file_list, rel_AGPM_pos = (y,x), coro = True, verbose = True, debug = False):
        """
        This method will find the location of the AGPM
        (roughly the location of the star) 
        """
        
        # add code for finding AGPM when coro = True, using relative position
        
        # if coro: 
            # find central pixel with frame_center from vip.var
            # then the position will be that plus the relative shift in y and x 
            
        ##### added by iain to fix AGPM location bug
        size_sci_sky_cube = open_fits(self.outpath + file_list[0]) # opens first sci/sky cube
        nz,ny,nx = size_sci_sky_cube.shape # gets size of it 
        median_all_cubes = np.zeros([len(file_list),ny,nx]) # makes empty array 
        for sc,fits_name in enumerate(file_list): # loops over all images 
            tmp = open_fits(self.outpath + fits_name) # opens the cube  
            median_all_cubes[sc] = tmp[-1] # takes the last entry (the median) and adds it to the empty array        
        ######
        #cube = open_fits(self.outpath + file_list[0])
        #nz, ny, nx = cube.shape
        median_frame = np.median(median_all_cubes, axis = 0)
        median_frame = frame_filter_lowpass(median_frame, median_size = 7, mode = 'median')       
        median_frame = frame_filter_lowpass(median_frame, mode = 'gauss',fwhm_size = 5)
        ycom,xcom = np.unravel_index(np.argmax(median_frame), median_frame.shape)
        if verbose:
            print('The location of the AGPM is','ycom =',ycom,'xcom =', xcom)
        if debug:
            pdb.set_trace()
        return [ycom, xcom]

class input_dataset():
    def __init__(self, inpath, outpath,coro= True): 
        self.inpath = inpath
        self.outpath = outpath
        old_list = os.listdir(self.inpath)
        self.file_list = [file for file in  old_list if file.endswith('.fits')]        
        self.dit_sci = fits_info.dit_sci
        self.ndit_sci = fits_info.ndit_sci
        self.ndit_sky = fits_info.ndit_sky
        self.dit_unsat = fits_info.dit_unsat
        self.ndit_unsat = fits_info.ndit_unsat
        self.dit_flat = fits_info.dit_flat

        
    def bad_columns(self, verbose = True, debug = False):
        """
        In NACO data there are systematic bad columns in the lower left quadrant
        This method will correct those bad columns with the median of the neighbouring pixels
        """
        #creating bad pixel map
        bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        for i in range(3, 509, 8):
            for j in range(512):
                bcm[j,i] = 1

        for fname in self.file_list:
            if verbose:
                print('Fixing', fname)
            tmp, header_fname = open_fits(self.inpath + fname,
                                                header = True, verbose = debug)
            if verbose:
                print(tmp.shape)
            #crop the bad pixel map to the same dimentions of the frames
            if len(tmp.shape) == 3:
                nz, ny, nx = tmp.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                for j in range(nz):
                    #replace bad columns in each frame of the cubes
                    tmp[j] = frame_fix_badpix_isolated(tmp[j],
                                    bpm_mask= bcm_crop, sigma_clip=3,
                                    num_neig=5, size=5, protect_mask=False,
                                    radius=30, verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify = 'fix')
                
            else:
                ny, nx = tmp.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                tmp = frame_fix_badpix_isolated(tmp,
                             bpm_mask= bcm_crop, sigma_clip=3, num_neig=5,
                             size=5, protect_mask=False, radius=30,
                             verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify = 'fix')
            if verbose:
                    print('done fixing',fname)

    def mk_dico(self, coro = True, verbose = True, debug = False):
        if coro:
           #creating a dictionary
           file_list = [f for f in listdir(self.outpath) if
                        isfile(join(self.outpath, f))]
           fits_list = []
           sci_list = []
           sci_list_mjd = []
           sky_list = []
           unsat_list = []
           unsat_list_mjd = []
           flat_list = []
           X_sci_list = []
           X_unsat_list = []
           flat_dark_list = []
           sci_dark_list = []
           unsat_dark_list = []
           sci_frames = []
           sky_frames = []
           unsat_frames = []
           flat_frames = []
           
           if verbose: 
               print('Creating dictionary')
           for fname in file_list:
               if fname.endswith('.fits') and fname.startswith('NACO'):
                   fits_list.append(fname)
                   cube, header = open_fits(self.outpath +fname, header=True,
                                            verbose=debug)
                   if header['HIERARCH ESO DPR CATG'] == 'SCIENCE'and \
                       header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_sci and\
                        cube.shape[0] > 2/3*min(self.ndit_sci): #avoid bad cubes
                            
                        sci_list.append(fname)
                        sci_list_mjd.append(header['MJD-OBS'])
                        X_sci_list.append(header['AIRMASS'])
                        sci_frames.append(cube.shape[0])
                        
                   elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                         header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                        header['HIERARCH ESO DET DIT'] == self.dit_sci and\
                        header['HIERARCH ESO DET NDIT'] in self.ndit_sky) and\
                       cube.shape[0] > 2/3*min(self.ndit_sky): #avoid bad cubes
                       sky_list.append(fname)
                       sky_frames.append(cube.shape[0])
                       
                   elif header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_unsat and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_unsat:
                       unsat_list.append(fname)
                       unsat_list_mjd.append(header['MJD-OBS'])
                       X_unsat_list.append(header['AIRMASS'])
                       unsat_frames.append(cube.shape[0])
                       
                   elif 'FLAT,SKY' in header['HIERARCH ESO DPR TYPE']:
                       flat_list.append(fname)
                       flat_frames.append(cube.shape[0])
                       
                   elif 'DARK' in header['HIERARCH ESO DPR TYPE']:
                       if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                           flat_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_sci:
                           sci_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_unsat:
                           unsat_dark_list.append(fname)
                           
           with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_list.txt", "w") as f:
                for sci in unsat_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_dark_list.txt", "w") as f:
                for sci in unsat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_dark_list.txt", "w") as f:
                for sci in flat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sci_dark_list.txt", "w") as f:
                for sci in sci_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_list.txt", "w") as f:
                for sci in flat_list:
                    f.write(sci+'\n')
           if verbose: 
               print('Done :)')


    def find_sky_in_sci_cube(self, nres = 3, coro = True, verbose = True, plot = None, debug = False):
       """
       Empty SKY list could be caused by a misclassification of the header in NACO data
       This method will check the flux of the SCI cubes around the location of the AGPM 
       A SKY cube should be less bright at that location allowing the seperation of cubes
       
       """
       if coro!=True:
           
       flux_list = []
       fname_list = []
       sci_list = []
       with open(self.outpath+"sci_list.txt", "r") as f: 
            tmp = f.readlines()
            for line in tmp:    
                sci_list.append(line.split('\n')[0])

       sky_list = []
       with open(self.outpath+"sky_list.txt", "r") as f: 
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])
        
       self.resel = (fits_info.wavelength*180*3600)/(fits_info.size_telescope *np.pi*
                                                 fits_info.pixel_scale)
                
       agpm_pos = find_AGPM_list(self, sci_list)
       if verbose: 
           print('The rough location of the star is','y  = ', agpm_pos[0] , 'x =', agpm_pos[1])

       #create the aperture
       circ_aper = CircularAperture((agpm_pos[1],agpm_pos[0]), round(nres*self.resel))

       #total flux through the aperture
       for fname in sci_list:
           cube_fname = open_fits(self.outpath + fname, verbose = debug) 
           median_frame = np.median(cube_fname, axis = 0)
           circ_aper_phot = aperture_photometry(median_frame,
                                                    circ_aper, method='exact')
       #append it to the flux list.
           circ_flux = np.array(circ_aper_phot['aperture_sum'])
           flux_list.append(circ_flux[0])
           fname_list.append(fname)
           if verbose: 
               print('centre flux has been measured for', fname)
       
       median_flux = np.median(flux_list)
       sd_flux = np.std(flux_list)

       if verbose:
           print('Sorting Sky from Sci')

       for i in range(len(flux_list)):
           if flux_list[i] < median_flux - 2*sd_flux:
               sky_list.append(fname_list[i])
               sci_list.remove(fname_list[i])
               symbol = 'bo'
           if plot: 
               if flux_list[i] > median_flux - 2*sd_flux:
                   symbol = 'go'
               else:
                   symbol = 'ro'
               plt.plot(i, flux_list[i]/median_flux , symbol)
       if plot:         
           plt.title('Normalised flux around star')
           plt.ylabel('normalised flux')
           plt.xlabel('cube')
           if plot == 'save':
               plt.savefig(self.outpath + 'flux_plot')
           if plot == 'show':
               plt.show()
                         
       with open(self.outpath+"sci_list.txt", "w") as f: 
                for sci in sci_list:
                    f.write(sci+'\n')
       with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')
       if verbose:
           print('done :)')
       
####### Iain's addition to find the derotation angles of the data ########

    def find_derot_angles(self, verbose=False):
        """ 
        For datasets with signification rotation when the telescope derotator is switched off. 
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
        
        #open the list of science images and add them to fits_list to be used in _derot_ang_ipag
        fits_list = []
        with open(self.outpath+"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:    
                fits_list.append(line.split('\n')[0])
        fits_list.sort()        
        
        def _derot_ang_ipag(self,fits_list=fits_list,loc='st'): 
            nsci = len(fits_list)
            parang = np.zeros(nsci)
            posang = np.zeros(nsci)
            rot_pt_off = np.zeros(nsci)
            n_frames_vec = np.ones(nsci, dtype=int)
            
            if loc == 'st':
                kw_par = 'HIERARCH ESO TEL PARANG START'
                kw_pos = 'HIERARCH ESO ADA POSANG'
            elif loc == 'nd':
                kw_par = 'HIERARCH ESO TEL PARANG END'
                kw_pos = 'HIERARCH ESO ADA POSANG END'        
            # FIRST COMPILE PARANG, POSANG and PUPILPOS 
            for ff in range(len(fits_list)):
                cube, header = open_fits(self.inpath+fits_list[ff], header=True, verbose=False)
                n_frames_vec[ff] = cube.shape[0]-1 # "-1" is because the last frame is the median of all others
                parang[ff] = header[kw_par]
                posang[ff] = header[kw_pos]
                pupilpos = 180.0 - parang[ff] + posang[ff]
                rot_pt_off[ff] = 90 + 89.44 - pupilpos
                if verbose:
                    print("parang: {}, posang: {}, rot_pt_off: {}".format(parang[ff],posang[ff],rot_pt_off[ff]))
               
            # NEXT CHECK IF THE OBSERVATION WENT THROUGH TRANSIT (change of sign in parang OR stddev of rot_pt_off > 1.)       
               
            rot_pt_off_med = np.median(rot_pt_off)
            rot_pt_off_std = np.std(rot_pt_off)    
            
            if np.min(parang)*np.max(parang) < 0. or rot_pt_off_std > 1.:
                if verbose:
                    print("The observation goes through transit and/or the pupil position was reset in the middle of the observation: ")
                    if np.min(parang)*np.max(parang) < 0.:
                        print("min/max parang: ", np.min(parang), np.max(parang))
                    if rot_pt_off_std > 1.:
                        print("the standard deviation of pupil positions is greater than 1: ", rot_pt_off_std)
                # find index where the transit occurs (change of sign of parang OR big difference in pupil pos)
                n_changes = 0
                for ff in range(len(fits_list)-1):
                    if parang[ff]*parang[ff+1] < 0. or np.abs(rot_pt_off[ff]-rot_pt_off[ff+1]) > 1.:
                        idx_transit = ff+1
                        n_changes+=1
                # check that these conditions only detected one passage through transit
                if n_changes != 1:
                    print(" {} passages of transit were detected (instead of 1!). Check that the input fits list is given in chronological order.".format(n_changes))
                    pdb.set_trace()
            
                rot_pt_off_med1 = np.median(rot_pt_off[:idx_transit])    
                rot_pt_off_med2 = np.median(rot_pt_off[idx_transit:])
                
                final_derot_angs = rot_pt_off_med1 - parang
                final_derot_angs[idx_transit:] = rot_pt_off_med2 - parang[idx_transit:]
            
            else:
                final_derot_angs = rot_pt_off_med - parang
        
            # MAKE SURE ANGLES ARE IN THE RANGE (-180,180)deg
            min_derot_angs = np.amin(final_derot_angs)
            nrot_min = min_derot_angs/360.
            if nrot_min < -0.5:
                final_derot_angs[np.where(final_derot_angs<-180)] = final_derot_angs[np.where(final_derot_angs<-180)] + np.ceil(nrot_min)*360.
            max_derot_angs = np.amax(final_derot_angs)
            nrot_max = max_derot_angs/360.
            if nrot_max > 0.5:
                final_derot_angs[np.where(final_derot_angs>180)] = final_derot_angs[np.where(final_derot_angs>180)] - np.ceil(nrot_max)*360.
                
            return -1.*final_derot_angs, n_frames_vec

        n_sci = len(fits_list)
        derot_angles_st, _ = _derot_ang_ipag(self,fits_list,loc='st')
        derot_angles_nd, n_frames_vec = _derot_ang_ipag(self,fits_list,loc='nd')
        final_derot_angs = np.zeros([n_sci,int(np.amax(n_frames_vec))])
        
        for sc in range(n_sci):
            n_frames = int(n_frames_vec[sc])
            nfr_vec = np.arange(n_frames)
            final_derot_angs[sc,:n_frames] = derot_angles_st[sc]+(((derot_angles_nd[sc]-derot_angles_st[sc])*nfr_vec/(n_frames-1)))
        write_fits(self.outpath+"derot_angles_uncropped.fits",final_derot_angs)    
        return final_derot_angs, n_frames_vec
        
        
        
        
        
