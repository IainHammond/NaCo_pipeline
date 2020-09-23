#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:54:43 2020

@author: lewis
"""

__author__ = 'Lewis Picker'
__all__ = ['dit_sci','ndit_sci','ndit_sky','dit_unsat','ndit_unsat','dit_flat','wavelength','size_telescope','pixel_scale']

#HD179218
#dit_sci = 0.35 #integration time for sci
#ndit_sci = [100] #number of frames per cube
#ndit_sky = [100]
#dit_unsat = 0.07 #integration time for unsaturated non coronagraphic images
#ndit_unsat = [400] #number of unsaturated frames in cubes

#HD206893
ndit_sci = [200] #number of frames per science cube
ndit_sky = [50] # number of frames per sky cube
ndit_unsat = [100] #number of frames in unsaturated cubes
dit_sci = 0.3 #integration time for science frames
dit_unsat = 0.1 #integration time for unsaturated non coronagraphic images
dit_flat = 0.2 #integration time for flat frames

wavelength = 3.8e-6 #meters
size_telescope = 8.2 #meters
pixel_scale = 0.02719  #arcsecs per pixel
