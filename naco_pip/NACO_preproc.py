#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs July 30 2020

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['calib_dataset']

import glob
import numpy as np
import os
import pathlib
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_shift, cube_recenter_via_speckles, cube_recenter_2dfit,frame_shift, cube_detect_badfr_correlation, cube_derotate

class calib_dataset():  #this class is for pre-processing of the calibrated data
    def __init__(self, inpath, outpath, final_sz, fwhm, coro = True):
        self.inpath = inpath
        self.outpath = outpath
        #self.resel = resel #resolution element in pixels (lambda/D). may just use fwhm
        self.final_sz = final_sz 
        self.fwhm = fwhm #open_fits(self.inpath+'fwhm.fits') 
        
    def recenter(self, method = '2dfit',model = '2gauss',nproc = 1, sigfactor = 4, subi_size = 21,verbose = True, debug = False, plot = False, coro = True):  
        """
        Recenters cropped science images by fitting a double Gaussian (negative+positive) to each median combined cube, and again by fitting a single negative Gaussian to the coronagraph using the speckle pattern of each median combined cube. 
        method: '2dfit' or 'speckle'
        model: '2gauss','gauss', 'moff'
        nproc: number of CPU cores
        sigfactor: float. If thresholding is performed, set the threshold in terms of gaussian sigma in the subimage (will depend on your cropping size)
        subi_size: int. Size of the square subimage sides in pixels.
        verbose: True for False, to provide extra information about the progress and results of the pipeline
        plot: True or False, set to False when running on M3
        coro: True for coronagraph data. False otherwise
        """  	
        if coro == False:
            if method != '2dfit':
                raise ValueError('Recentering method invalid')
            if model == '2gauss':
                raise ValueError('2Gaus requires coronagraphic data')

	#get all the science cubes into a list
        self.sci_list = []
        with open(self.inpath +"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                self.sci_list.append(line.split('\n')[0])
        
        if verbose:
        	print(len(self.sci_list),'science cubes')
        
        ncubes = len(self.sci_list)      
        
        # Creates a master science cube with just the median of each cube
        for sc, fits_name in enumerate(self.sci_list): # enumerate over the list of all science cubes
            tmp = open_fits(self.inpath+fits_name, verbose=verbose) #open cube as tmp
            if sc == 0: 
                self.ndit, ny, nx = tmp.shape #dimensions of cube
                tmp_tmp = np.zeros([ncubes,ny,nx]) # template cube with the median of each SCI cube. np.zeros is array filled with zeros
            tmp_tmp[sc]= np.median(tmp, axis=0) # median frame of cube tmp 
          
        if method == 'speckle':
                # FOR GAUSSIAN		
                #registered science sube, low+high pass filtered cube,cube with stretched values, x shifts, y shifts, optimal inner radius value when you fit an annulus
                tmp_tmp,cube_sci_lpf,cube_stret,sx, sy,opt_rad = cube_recenter_via_speckles(tmp_tmp, cube_ref=None,
				                                                alignment_iter = 5, gammaval = 1,
				                                                min_spat_freq = 0.5, max_spat_freq = 3,
				                                                fwhm = self.fwhm, debug = debug,
				                                                recenter_median = True, negative = coro,
				                                                fit_type='gaus', crop=False,subframesize = final_sz, 
				                                                imlib='opencv',interpolation='lanczos4',plot=plot,full_output=True)	
        elif method == '2dfit':	
                # DOUBLE GAUSSIAN          	
                params_2g = {'fwhm_neg': 0.8*self.fwhm, 'fwhm_pos': 2*self.fwhm, 'theta_neg': 48., 'theta_pos':135., 'neg_amp': 0.8}
                res = cube_recenter_2dfit(tmp_tmp, xy=None, fwhm=self.fwhm, subi_size=subi_size,
				                      model=model, nproc=nproc, imlib='opencv', 
				                      interpolation='lanczos4', offset=None, 
				                      negative=False, threshold=True, sigfactor=sigfactor, 
				                      fix_neg=False, params_2g=params_2g,
				                      save_shifts=False, full_output=True, verbose=verbose,
				                      debug=debug, plot=plot)
                sy = res[1]
                sx = res[2]	                              
	#	true_agpm_cen = (res[4][0],res[3][0])
	#	true_fwhm_pos = (res[5][0],res[6][0])
	#	true_fwhm_neg = (res[7][0],res[8][0])
	#	true_theta_pos = res[9][0]
	#	true_theta_neg = res[10][0]
	#	amp_pos = res[11][0]
	#	amp_neg = res[12][0]
	#	true_neg_amp = amp_neg/amp_pos
	#	params_2g = {'fwhm_neg': true_fwhm_neg, 'fwhm_pos': true_fwhm_pos, 
	#		                 'theta_neg': true_theta_neg, 'theta_pos':true_theta_pos, 
	#		                 'neg_amp': true_neg_amp}
	#	# second: fixing params for neg gaussian - applied on individual frames. returns recentered array, and x-y shifts             
	#	tmp_tmp, sy, sx = cube_recenter_2dfit(tmp_tmp, xy=true_agpm_cen, 
	#		                                        fwhm=self.fwhm, subi_size=subi_size, 
	#		                                        model=model, nproc=nproc, imlib='opencv', 
	#		                                        interpolation='lanczos4',
	#		                                        offset=None, negative=False, 
	#		                                        threshold=True, sigfactor=sigfactor, 
	#		                                        fix_neg=True, params_2g=params_2g,
	#		                                        save_shifts=False, full_output=True, 
	#		                                        verbose=verbose, debug=debug, plot=plot)		
   		
	# Load original cubes, shift them, and create master cube
        tmp_tmp = np.zeros([int(self.ndit*ncubes),ny,nx]) #np.zeros makes an array full of zeros. we dont need our old tmp_tmp anymore		   

        for sc, fits_name in enumerate(self.sci_list):
            tmp = open_fits(self.inpath+fits_name, verbose=verbose) #opens original cube 
            for dd in range(self.ndit):         
                tmp_tmp[int(sc*self.ndit+dd)] = frame_shift(tmp[dd],shift_y=sy[sc],shift_x=sx[sc],imlib='opencv')

        pathlib.Path(self.outpath).mkdir(parents=True, exist_ok=True)     
        
	# write all the shifts
        write_fits(self.outpath+'x_shifts_{}_{}.fits'.format(method,model), sx) # writes the x shifts to the file
        write_fits(self.outpath+'y_shifts_{}_{}.fits'.format(method,model), sy) # writes the y shifts to the file
        write_fits(self.outpath+"master_cube_{}_{}.fits".format(method,model),tmp_tmp) #makes the master cube
		
        if verbose:
                print('Shifts applied, master cube saved')	     
        
		
    def bad_frame_removal(self, recenter_method, recenter_model, correlation_thres = 0.9, pxl_shift_thres = 0.5, crop_size = 31, verbose = True, debug = False, plot = False):	
        """
       For removing outlier frames often caused by AO errors. To be run after recentering is complete. Takes the recentered mastercubes and removes frames with a shift greater than a user defined pixel threshold in x or y above the median shift. It then takes the median of those cubes and correlates them to the median combined mastercube. A threshold, also set by the user, removes all those cubes below the threshold from the mastercube and rotation file, then saves both as new files
        """     
        
        if verbose:
            print('\n')
            print('Beginning bad frame removal...')
            print('\n')
        angle_file = open_fits(self.inpath+'derot_angles.fits') #opens the rotation file
        recentered_cube = open_fits(self.outpath+"master_cube_{}_{}.fits".format(recenter_method,recenter_model)) # loads the master cube       
	    
        #open x shifts file for the respective method
        x_shifts = open_fits(self.outpath+"x_shifts_{}_{}.fits".format(recenter_method,recenter_model))
        median_sx = np.median(x_shifts) #median of x shifts 

        #opens y shifts file for the respective method
        y_shifts = open_fits(self.outpath+"y_shifts_{}_{}.fits".format(recenter_method,recenter_model))
        median_sy = np.median(y_shifts) #median of y shifts    
	    
        x_shifts_long = np.zeros([len(self.sci_list)*self.ndit]) # list with number of cubes times number of frames in each cube as the length
        y_shifts_long = np.zeros([len(self.sci_list)*self.ndit])

        for i in range(len(self.sci_list)):
                x_shifts_long[i*self.ndit:(i+1)*self.ndit] = x_shifts[i] # sets the average shifts of all frames in a cube 
                y_shifts_long[i*self.ndit:(i+1)*self.ndit] = y_shifts[i]

        if debug:
                write_fits(self.outpath+'x_shifts_long_{}_{}.fits'.format(recenter_method,recenter_model),x_shifts_long) # saves shifts to file
                write_fits(self.outpath+'y_shifts_long_{}_{}.fits'.format(recenter_method,recenter_model),y_shifts_long)   
        x_shifts = x_shifts_long
        y_shifts = y_shifts_long 
	    
        if verbose:
                print("x shift median:",median_sx)
                print("y shift median:",median_sy)
                print('Running pixel shift check...')
		    
        bad = []
        good = []
	    
        # this system works, however it can let a couple of wild frames slip through at a 0.5 pixel threshold. may need a stricter threshold depending on the data
        i = 0 
        shifts = list(zip(x_shifts,y_shifts))
        for sx,sy in shifts: #iterate over the shifts to find any greater or less than pxl_shift_thres pixels from median
                if abs(sx) < ((abs(median_sx)) + pxl_shift_thres) and abs(sx) > ((abs(median_sx)) - pxl_shift_thres) and abs(sy) < ((abs(median_sy)) + pxl_shift_thres) and abs(sy) > ((abs(median_sy)) - pxl_shift_thres):
                    good.append(i)
                else:   		
                    bad.append(i)
                i=i+1    
	    
        # only keeps the files that weren't shifted above the threshold
        frames_pxl_threshold = recentered_cube[good]
        if verbose:
            print('Frames within pixel shift threshold:',len(recentered_cube[good]))		
        # only keeps the corresponding derotation entry for the frames that were kept
        angle_pxl_threshold = angle_file[good]
	    
        #makes array of good and bad frames from the recentered mastercube		                                              
        tmp_median = np.median(frames_pxl_threshold, axis=0)  #median frame of cube of remaining frames
	
        if verbose:
            print('Running frame correlation check...')    
	
        good_frames, bad_frames = cube_detect_badfr_correlation(frames_pxl_threshold,
			                                                frame_ref = tmp_median,
			                                                crop_size=crop_size, 
			                                                dist='pearson',
			                                                threshold=correlation_thres,                                                                    
			                                                plot=plot,
			                                                verbose=verbose)
	    
        if verbose:
            print('Kept',len(good_frames),'frames out of',len(self.sci_list)*self.ndit)
	    
        #only keeps the files that were above the correlation threshold
        frames_threshold = frames_pxl_threshold[good_frames]
	    
        if verbose:
            print('Frames within correlation threshold:',len(frames_pxl_threshold[good_frames]))	    
        # only keeps the derotation entries for the good frames above the correlation threshold     
        angle_threshold = angle_pxl_threshold[good_frames]
	    
        # saves the good frames to a new file, and saves the derotation angles to a new file
        write_fits(self.outpath+'master_cube_{}_{}_good_frames.fits'.format(recenter_method,recenter_model), frames_threshold)  
        write_fits(self.outpath+'derot_angles_{}_{}.fits'.format(recenter_method,recenter_model), angle_threshold)  
        if verbose: 
                print('Saved good frames and their respective rotations to file')	
 
