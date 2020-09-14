#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs July 30 2020

@author: Iain
"""
__author__ = 'Iain Hammond'
__all__ = ['calib_dataset']

import pdb
import numpy as np
import pyprind
import os
import pathlib
from matplotlib import pyplot as plt
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_recenter_via_speckles, cube_recenter_2dfit,frame_shift, cube_detect_badfr_correlation, cube_crop_frames
from naco_pip import fits_info

class calib_dataset():  #this class is for pre-processing of the calibrated data
    def __init__(self, inpath, outpath, final_sz, recenter_method, recenter_model, coro = True):
        self.inpath = inpath
        self.outpath = outpath
        self.final_sz = final_sz 
        fwhm = open_fits(self.inpath+'fwhm.fits',verbose=True)[0] # changed this to open the file as sometimes we wont run get_stellar_psf() or it may have already run. fwhm is first entry in the file
        self.derot_angles_cropped = open_fits(self.inpath+'derot_angles_cropped.fits',verbose=True)
        self.recenter_method = recenter_method
        self.recenter_model = recenter_model
        self.sci_list = []
        #get all the science cubes into a list
        with open(self.inpath +"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                self.sci_list.append(line.split('\n')[0])
        self.sci_list.sort() # make sure they are in order so derotation doesn't make a mess of the frames
        self.real_ndit_sci = [] 
        for sc, fits_name in enumerate(self.sci_list): # enumerate over the list of all science cubes
            tmp = open_fits(self.inpath+'4_sky_subtr_imlib_'+fits_name, verbose=False)
            self.real_ndit_sci.append(tmp.shape[0]) # gets length of each cube for later use       
        
        
    def recenter(self, nproc = 1, sigfactor = 4, subi_size = 21, verbose = True, debug = False, plot = False, coro = True):  
        """
        Recenters cropped science images by fitting a double Gaussian (negative+positive) to each median combined cube, or by fitting a single negative Gaussian to the coronagraph using the speckle pattern of each median combined cube. 
        
        Parameters:
        ***********  
        sigfactor: float. 
            If thresholding is performed, set the threshold in terms of gaussian sigma in the subimage (will depend on your cropping size)
        subi_size: int. 
            Size of the square subimage sides in pixels.
        verbose: True for False
            To provide extra information about the progress and results of the pipeline
        plot: True or False
            Set to False when running on M3
        coro: True for coronagraph data. False otherwise
        
        Writes fits to file:
        ***********  
        x_shifts_{recenter_method}_{recenter_model}.fits # writes the x shifts to the file
        y_shifts_{recenter_method}_{recenter_model}.fits # writes the y shifts to the file
        master_cube_{recenter_method}_{recenter_model}.fits # makes the master cube
        derot_angles_1d.fits # makes a vector of derotation angles
        """  	
        
        if coro == False:
            if self.recenter_method != '2dfit':
                raise ValueError('Recentering method invalid')
            if self.recenter_model == '2gauss':
                raise ValueError('2Gauss requires coronagraphic data')        
        if verbose:
        	print(len(self.sci_list),'science cubes')
        	print(self.sci_list)
        
        ncubes = len(self.sci_list)     
         
        fwhm_all = open_fits(self.inpath+'fwhm.fits',verbose=debug) # changed this to open the file as sometimes we wont run get_stellar_psf() or it may have already run   
        fwhm = fwhm_all[0] # fwhm is the first entry in the file 
        fwhm = fwhm.item() # changes from numpy.float32 to regular float so it will work in VIP
        if verbose:
            print('fwhm:',fwhm,'of type',type(fwhm)) 
        
        # Creates a master science cube with just the median of each cube
        if verbose:        
            print('Creating master science cube (median of each science cube)....')            
            
        bar = pyprind.ProgBar(len(self.sci_list), stream=1)    
        for sc, fits_name in enumerate(self.sci_list): # enumerate over the list of all science cubes
            tmp = open_fits(self.inpath+'4_sky_subtr_imlib_'+fits_name, verbose=debug) #open cube as tmp            
            if sc == 0: 
                self.ndit, ny, nx = tmp.shape #dimensions of cube
                tmp_tmp = np.zeros([ncubes,ny,nx]) # template cube with the median of each SCI cube. np.zeros is array filled with zeros
            tmp_tmp[sc]= np.median(tmp, axis=0) # median frame of cube tmp 
            bar.update()         
        
        if self.recenter_method == 'speckle':
                # FOR GAUSSIAN
                print('##### Recentering via speckle pattern #####')
                #registered science sube, low+high pass filtered cube,cube with stretched values, x shifts, y shifts, optimal inner radius value when you fit an annulus
                tmp_tmp,cube_sci_lpf,cube_stret,sx, sy,opt_rad = cube_recenter_via_speckles(tmp_tmp, cube_ref=None,
				                                                alignment_iter = 5, gammaval = 1,
				                                                min_spat_freq = 0.5, max_spat_freq = 3,
				                                                fwhm = fwhm, debug = debug,
				                                                recenter_median = True, negative = coro,
				                                                fit_type='gaus', crop=False, subframesize = subi_size, 
				                                                imlib='opencv',interpolation='lanczos4',plot = plot, full_output = True)	
        elif self.recenter_method == '2dfit':	
                # DOUBLE GAUSSIAN
                print('##### Recentering via 2dfit #####')         	
                params_2g = {'fwhm_neg': 0.8*fwhm, 'fwhm_pos': 2*fwhm, 'theta_neg': 48., 'theta_pos':135., 'neg_amp': 0.8}
                res = cube_recenter_2dfit(tmp_tmp, xy=None, fwhm=fwhm, subi_size=subi_size,
				                      model=self.recenter_model, nproc=nproc, imlib='opencv', 
				                      interpolation='lanczos4', offset=None, 
				                      negative=False, threshold=True, sigfactor=sigfactor, 
				                      fix_neg=False, params_2g=params_2g,
				                      save_shifts=False, full_output=True, verbose=verbose,
				                      debug=debug, plot=plot)
                sy = res[1]
                sx = res[2]	                              
#                true_agpm_cen = (res[4][0],res[3][0])
#                true_fwhm_pos = (res[5][0],res[6][0])
#                true_fwhm_neg = (res[7][0],res[8][0])
#                true_theta_pos = res[9][0]
#                true_theta_neg = res[10][0]
#                amp_pos = res[11][0]
#                amp_neg = res[12][0]
#                true_neg_amp = amp_neg/amp_pos
#                params_2g = {'fwhm_neg': true_fwhm_neg, 'fwhm_pos': true_fwhm_pos, 
#			                 'theta_neg': true_theta_neg, 'theta_pos':true_theta_pos, 
#			                 'neg_amp': true_neg_amp}
#		# second: fixing params for neg gaussian - applied on individual frames. returns recentered array, and x-y shifts             
#                tmp_tmp, sy, sx = cube_recenter_2dfit(tmp_tmp, xy=true_agpm_cen, 
#			                                        fwhm=self.fwhm, subi_size=subi_size, 
#			                                        model=model, nproc=nproc, imlib='opencv', 
#			                                        interpolation='lanczos4',
#			                                        offset=None, negative=False, 
#			                                        threshold=True, sigfactor=sigfactor, 
#			                                        fix_neg=True, params_2g=params_2g,
#			                                        save_shifts=False, full_output=True, 
#			                                        verbose=verbose, debug=debug, plot=plot)		
   		# LOAD IN REAL_NDIT_SCI   		
	    # Load original cubes, shift them, and create master cube
        tmp_tmp = np.zeros([int(np.sum(self.real_ndit_sci)),ny,nx]) #makes an array full of zeros, length of the sum of each entry in the sci dimensions file. we dont need our old tmp_tmp anymore		   
        angles_1dvector = np.zeros([int(np.sum(self.real_ndit_sci))]) # makes empty array for derot angles, length of number of frames 
        for sc, fits_name in enumerate(self.sci_list):
            tmp = open_fits(self.inpath+'4_sky_subtr_imlib_'+fits_name, verbose=debug) #opens science cube
            dim = int(self.real_ndit_sci[sc]) #gets the integer dimensions of this science cube
            for dd in range(dim): #dd goes from 0 to the largest dimension
                tmp_tmp[int(np.sum(self.real_ndit_sci[:sc]))+dd] = frame_shift(tmp[dd],shift_y=sy[sc],shift_x=sx[sc],imlib='opencv') #this line applies the shifts to all the science images in the cube the loop is currently on. it also converts all cubes to a single long cube by adding the first dd frames, then the next dd frames from the next cube and so on
                angles_1dvector[int(np.sum(self.real_ndit_sci[:sc]))+dd] = self.derot_angles_cropped[sc][dd] # turn 2d rotation file into a vector here same as for the mastercube above
                # sc*ndit+dd i don't think this line works for variable sized cubes 
        pathlib.Path(self.outpath).mkdir(parents=True, exist_ok=True)
        
	# write all the shifts
        write_fits(self.outpath+'x_shifts_{}_{}.fits'.format(self.recenter_method,self.recenter_model), sx) # writes the x shifts to the file
        write_fits(self.outpath+'y_shifts_{}_{}.fits'.format(self.recenter_method,self.recenter_model), sy) # writes the y shifts to the file
        write_fits(self.outpath+"master_cube_{}_{}.fits".format(self.recenter_method,self.recenter_model),tmp_tmp) #makes the master cube
        write_fits(self.outpath+'derot_angles_1d.fits',angles_1dvector) # writes the 1D array of derotation angles
        if verbose:
            print('Shifts applied, master cube saved')	     
     
    def bad_frame_removal(self, correlation_thres = 0.9, pxl_shift_thres = 0.5, crop_size = 31, verbose = True, debug = False, plot = 'save'):	
        """
       For removing outlier frames often caused by AO errors. To be run after recentering is complete. Takes the recentered mastercubes and removes frames with a shift greater than a user defined pixel threshold in x or y above the median shift. It then takes the median of those cubes and correlates them to the median combined mastercube. A threshold, also set by the user, removes all those cubes below the threshold from the mastercube and rotation file, then saves both as new files for use in post processing
        recenter_method: string, same as method used to recenter the frames
        recenter_model: string, same as the model used to recenter the frames
        correlation_thres: default is 0.9, can take any value between 0.0 and 1.0. Any frames below this correlation threshold will be excluded. Generally recommended to use a value between 0.90 and 0.95 (90-95% correlation to the median), however depends on crop_size
        pxl_shift_thres: decimal, in units of pixels. Default is 0.5 pixels. Any shifts in the x or y direction greater than this threshold will cause the frame/s to be labelled as bad and thus removed 
        crop_size: integer, must be odd. Default is 31. This sets the cropping during frame correlation to the median    
        plot: 'save' to write to file the correlation plot, None will not     
        """     
        
        if verbose:
            print('\n')
            print('Beginning bad frame removal...')
            print('\n')
        angle_file = open_fits(self.outpath+'derot_angles_1d.fits') #opens the rotation file
        recentered_cube = open_fits(self.outpath+"master_cube_{}_{}.fits".format(self.recenter_method,self.recenter_model)) # loads the master cube       
	    
        #open x shifts file for the respective method
        x_shifts = open_fits(self.outpath+"x_shifts_{}_{}.fits".format(self.recenter_method,self.recenter_model))
        median_sx = np.median(x_shifts) #median of x shifts 

        #opens y shifts file for the respective method
        y_shifts = open_fits(self.outpath+"y_shifts_{}_{}.fits".format(self.recenter_method,self.recenter_model))
        median_sy = np.median(y_shifts) #median of y shifts    
	    
        #x_shifts_long = np.zeros([len(self.sci_list)*self.ndit]) # list with number of cubes times number of frames in each cube as the length
        #y_shifts_long = np.zeros([len(self.sci_list)*self.ndit])
        x_shifts_long = np.zeros([int(np.sum(self.real_ndit_sci))])
        y_shifts_long = np.zeros([int(np.sum(self.real_ndit_sci))])

        for i in range(len(self.sci_list)): # from 0 to the length of sci_list
            ndit = self.real_ndit_sci[i] # gets the dimensions of the cube
            x_shifts_long[i*ndit:(i+1)*ndit] = x_shifts[i] # sets the average shifts of all frames in that cube 
            y_shifts_long[i*ndit:(i+1)*ndit] = y_shifts[i]

        if verbose:
            write_fits(self.outpath+'x_shifts_long_{}_{}.fits'.format(self.recenter_method,self.recenter_model),x_shifts_long) # saves shifts to file
            write_fits(self.outpath+'y_shifts_long_{}_{}.fits'.format(self.recenter_method,self.recenter_model),y_shifts_long)   
        x_shifts = x_shifts_long
        y_shifts = y_shifts_long 
	    
        if verbose:
            print("x shift median:",median_sx)
            print("y shift median:",median_sy)
            print('Running pixel shift check...')
		    
        bad = []
        good = []
	    
        # this system works, however it can let a couple of wild frames slip through at a 0.5 pixel threshold. may need a stricter threshold depending on the data and crop size
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
	    
        if verbose:
	        print('########### Median combining {} frames for correlation check... ###########'.format(len(frames_pxl_threshold)))
	    
        #makes array of good frames from the recentered mastercube		                                              
        tmp_median = np.median(frames_pxl_threshold, axis=0)  #median frame of remaining frames
	
        if verbose:
            print('Running frame correlation check...')
	
        good_frames, bad_frames = cube_detect_badfr_correlation(frames_pxl_threshold,
			                                                frame_ref = tmp_median,
			                                                crop_size=crop_size, 
			                                                dist='pearson',
			                                                threshold=correlation_thres,                                                                    
			                                                plot=True,
			                                                verbose=verbose) 
        if plot =='save':
            plt.savefig(self.outpath+'frame_correlation.pdf')
	    
        #only keeps the files that were above the correlation threshold
        frames_threshold = frames_pxl_threshold[good_frames]
	    
        if verbose:
            print('Frames within correlation threshold:',len(frames_pxl_threshold[good_frames]))  
        # only keeps the derotation entries for the good frames above the correlation threshold     
        angle_threshold = angle_pxl_threshold[good_frames]
	    
        # saves the good frames to a new file, and saves the derotation angles to a new file
        write_fits(self.outpath+'master_cube_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model), frames_threshold)  
        write_fits(self.outpath+'derot_angles_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model), angle_threshold)  
        if verbose: 
            print('Saved good frames and their respective rotations to file')	
 
    def crop_cube(self, arcsecond_diameter = 3, verbose = True, full_output= False):
    
        """Crops frames in the master cube after recentering and bad frame removal. Recommended for post-processing.

        Parameters
        ----------
        arcsecond_diameter : float or int
            Size of the frames diameter in arcseconds. Default of 3" for NaCO corresponds to 111x111 (x,y) pixel frames

        verbose : bool optional
            If True extra messages of completion are showed.

        Writes to fits file
        -------
        cropped cube : numpy ndarray
            Cube with cropped frames.

        """
        if os.path.isfile(self.outpath+'master_cube_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model)) == False:
            raise NameError('Missing master cube from recentering and bad frame removal!')            
        
        crop_size = int((arcsecond_diameter)/(fits_info.pixel_scale))        
     
        master_cube = open_fits(self.outpath+'master_cube_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model),verbose = verbose)
             
        if verbose:
            print('######### Running frame cropping #########')       
            
        cropped_cube = cube_crop_frames(master_cube, crop_size, force = False, verbose = verbose, full_output = full_output)
        
        write_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped.fits'.format(self.recenter_method,self.recenter_model), cropped_cube) 
         
    def median_binning(self, binning_factor = 10, verbose = True):
        """ 
        Median combines the frames within the master science cube as per the binning factor, and makes the necessary changes to the derotation file
        
        Parameters:
        ***********        

        binning_factor: int,list or tuple
            Defines how many frames to median combine. Use a list or tuple to run binning multiple times with different factors. Default = 10
                  
        Writes to fits file:
        ********
        master_cube_binned: the binned master cube
        derot_angles_binned: the binned derotation angles
             
        """
        
        if os.path.isfile(self.outpath+'master_cube_{}_{}_good_frames_cropped.fits'.format(self.recenter_method,self.recenter_model)) == False:
            raise NameError('Missing master cube from recentering and bad frame removal!') 
            
        if os.path.isfile(self.outpath+'derot_angles_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model)) == False:
            raise NameError('Missing derotation angles files from recentering and bad frame removal!') 
        
        master_cube = open_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped.fits'.format(self.recenter_method,self.recenter_model), verbose = verbose)
        derot_angles = open_fits(self.outpath+'derot_angles_{}_{}_good_frames.fits'.format(self.recenter_method,self.recenter_model), verbose = verbose)
        
        if isinstance(binning_factor, int) == False and isinstance(binning_factor,list) == False and isinstance(binning_factor,tuple) == False: # if it isnt int, tuple or list then raise an error
            raise TypeError('Invalid binning_factor! Use either int, list or tuple')
    
        if isinstance(binning_factor, int):
            if binning_factor == 1: # will skip binning if it's 1
                print('Binning factor is 1 (cant bin any frames). Skipping binning...')
                write_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), master_cube)
                write_fits(self.outpath+'derot_angles_{}_{}_good_frames_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), derot_angles)
            else:
                print('##### Median binning frames with binning_factor = {} #####'.format(binning_factor))                               
                nframes,ny,nx = master_cube.shape
                
                # defines a new empty array of length frames divided by the binning factor. fills it with median combined frames based on the binning factor. The last entry will have the median of the no. of frames equal to the binning factor + left over frames at the end. then makes changes to the derot angles files (median combined the respective entries)       
                
                derot_angles_binned = np.zeros([int(nframes/binning_factor)])
                master_cube_binned = np.zeros([int(nframes/binning_factor),ny,nx])
               
                for idx,frame in enumerate(range(binning_factor,nframes,binning_factor)):
                    if idx == (int(nframes/binning_factor)-1):
                        master_cube_binned[idx] = np.median(master_cube[frame-binning_factor:])
                        derot_angles_binned[idx] = np.median(derot_angles[frame-binning_factor:])
                    master_cube_binned[idx] = np.median(master_cube[frame-binning_factor:frame],axis=0)
                    derot_angles_binned[idx] = np.median(derot_angles[frame-binning_factor:frame])
                
                write_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor),master_cube_binned)
                write_fits(self.outpath+'derot_angles_{}_{}_good_frames_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), derot_angles_binned)
                
        if isinstance(binning_factor,list) or isinstance(binning_factor,tuple):
            for binning_factor in binning_factor:
                if binning_factor == 1: # doesn't bin with 1 but will loop over the other factors in the list or tuple
                    print('Binning factor is 1 (cant bin any frames). Skipping binning...')
                    write_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), master_cube)
                    write_fits(self.outpath+'derot_angles_{}_{}_good_frames_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), derot_angles)                
                else:
                    print('##### Median binning frames with binning_factor = {} #####'.format(binning_factor))
                    nframes,ny,nx = master_cube.shape
                    derot_angles_binned = np.zeros([int(nframes/binning_factor)])
                    master_cube_binned = np.zeros([int(nframes/binning_factor),ny,nx])
                   
                    for idx,frame in enumerate(range(binning_factor,nframes,binning_factor)):
                        if idx == (int(nframes/binning_factor)-1):
                            master_cube_binned[idx] = np.median(master_cube[frame-binning_factor:])
                            derot_angles_binned[idx] = np.median(derot_angles[frame-binning_factor:])
                        master_cube_binned[idx] = np.median(master_cube[frame-binning_factor:frame],axis=0)
                        derot_angles_binned[idx] = np.median(derot_angles[frame-binning_factor:frame])
                    
                    write_fits(self.outpath+'master_cube_{}_{}_good_frames_cropped_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor),master_cube_binned)
                    write_fits(self.outpath+'derot_angles_{}_{}_good_frames_bin{}.fits'.format(self.recenter_method,self.recenter_model,binning_factor), derot_angles_binned)
