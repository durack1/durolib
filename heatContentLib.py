# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:38:01 2014

@author: durack1
"""

import gc
import cdms2 as cdm
import cdutil as cdu
import numpy as np
import MV2 as mv
import seawater as sw ; # was seawater.csiro
#from matplotlib import pyplot as plt
from numpy import isnan,tile,transpose
from scipy import interpolate
#from matplotlib.cm import RdBu_r

np.seterr(all='ignore') ; # Cautious use of this turning all error reporting off - shouldn't be an issue as using masked arrays

# Set netcdf file criterion - turned on from default 0s
cdm.setCompressionWarnings(0) ; # Suppress warnings
cdm.setNetcdfShuffleFlag(0)
cdm.setNetcdfDeflateFlag(1)
cdm.setNetcdfDeflateLevelFlag(9)
# Hi compression: 1.4Gb file ; # Single salt variable
# No compression: 5.6Gb ; Standard (compression/shuffling): 1.5Gb ; Hi compression w/ shuffling: 1.5Gb
cdm.setAutoBounds(1) ; # Ensure bounds on time and depth axes are generated

#%%
# Purge spyder variables
if 'e' in locals():
    del(e,pi,sctypeNA,typeNA)
    gc.collect()
#%%

# Define functions
def maskFill(mask,points,index):
    interpolant = interpolate.LinearNDInterpolator(points[index,:],np.array(mask.flatten())[index]) ; # Create interpolant
    maskFilled = interpolant(points[:,0].squeeze(),points[:,1].squeeze()) ; # Use interpolant to create filled matrix    
    return maskFilled


def scrubNaNAndMask(var,maskVar):
    # Check for NaNs
    nanvals = isnan(var)
    var[nanvals] = 1e+20
    var = mv.masked_where(maskVar>=1e+20,var)
    var = mv.masked_where(maskVar.mask,var)
    return var


def makeHeatContent(salt,temp,destMask,thetao,pressure):
    # Remap variables to short names
    #print salt.getAxisIds()
    s       = salt(squeeze=1)
    #print s.getAxisIds()
    t       = temp(squeeze=1)
    mask    = destMask
    #print mask.getAxisIds()
    del(salt,temp,destMask) ; gc.collect()
    depthInd = 0 ; # Set depth index
    
    #print 's:    ',s.min(),s.max()
    #print 't:    ',t.min(),t.max()

    # Fix out of bounds values
    t = mv.where(t<-2.6,-2.6,t) ; # Fix for NaN values    
    
    # Calculate pressure - inputs depth & lat
    # Create z-coordinate from salinity input
    if not pressure:
        zCoord                         = s.getAxis(depthInd) ; # Assume time,depth,latitude,longitude grid
        yCoord                         = s.getAxis(depthInd+1)
        yCoord                         = tile(yCoord,(s.shape[depthInd+2],1)).transpose()
        depthLevels                    = tile(zCoord.getValue(),(s.shape[depthInd+2],s.shape[depthInd+1],1)).transpose()
        pressureLevels                 = sw.pres(np.array(depthLevels),np.array(yCoord))
        del(zCoord,yCoord,depthLevels) ; gc.collect()
    else:
        pressureLevels                 = s.getAxis(depthInd)
        #print pressureLevels.getValue()
        pressureLevels                 = transpose(tile(pressureLevels,(s.shape[depthInd+2],s.shape[depthInd+1],1)))
    pressureLevels                 = cdm.createVariable(pressureLevels,id='pressureLevels')
    pressureLevels.setAxis(0,s.getAxis(depthInd))
    pressureLevels.setAxis(1,s.getAxis(depthInd+1))
    pressureLevels.setAxis(2,s.getAxis(depthInd+2))
    pressureLevels.units_long      = 'decibar (pressure)'
    pressureLevels.positive        = 'down'
    pressureLevels.long_name       = 'sea_water_pressure'
    pressureLevels.standard_name   = 'sea_water_pressure'
    pressureLevels.units           = 'decibar'
    pressureLevels.axis            = 'Z'
    
    #print 'pres: ',pressureLevels.min(),pressureLevels.max()
    #print pressureLevels.shape
    
    #print s.shape
    #print t.shape
    #print mask.shape

    # Calculate temp,rho,cp - inputs temp,salt,pressure
    if thetao:
        # Process potential temperature to in-situ
        temp        = sw.temp(np.array(s),np.array(t),np.array(pressureLevels)); # units degrees C
    rho             = sw.dens(np.array(s),np.array(temp),np.array(pressureLevels)) ; # units kg m-3
    cp              = sw.cp(np.array(s),np.array(temp),np.array(pressureLevels)) ; # units J kg-1 C-1

    # Correct instances of NaN values and fix masks - applied before cdms variables are created otherwise names/ids/attributes are reset
    temp            = scrubNaNAndMask(temp,s)
    rho             = scrubNaNAndMask(rho,s)
    cp              = scrubNaNAndMask(cp,s)
    
    #print 'temp: ',temp.min(),temp.max()    
    #print 'rho:  ',rho.min(),rho.max()
    #print 'cp:   ',cp.min(),cp.max()
    
    # Calculate heatContent - inputs temp,rho,cp
    heatContent     = np.array(temp)*np.array(rho)*np.array(cp) ; # units J
    
    # Correct instances of NaN values and fix masks - applied before cdms variables are created otherwise names/ids/attributes are reset
    heatContent     = scrubNaNAndMask(heatContent,s)    

    #print 'hc:   ',heatContent.min(),heatContent.max()

    # Interpolate to standard levels - inputs heatContent,levels
    newDepth        = np.array([5,10,20,30,40,50,75,100,125,150,200,300,500,700,1000,1500,1800,2000]).astype('f');
    newDepth_bounds = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,75],[75,100],[100,125],[125,150],
                                [150,200],[200,300],[300,500],[500,700],[700,1000],[1000,1500],[1500,1800],[1800,2000]]).astype('f')
    # Interpolate to standard levels
    #print heatContent.shape
    #print heatContent.getAxisIds()
    #print pressureLevels.shape
    #print pressureLevels.getAxisIds()
    
    # Reset variable axes
    heatContent.setAxis(0,s.getAxis(0))
    heatContent.setAxis(1,s.getAxis(1))
    heatContent.setAxis(2,s.getAxis(2))

    #print heatContent.shape
    #print heatContent.getAxisIds()
    #print pressureLevels.shape
    #print pressureLevels.getAxisIds()
    
    heatContent_depthInterp     = cdu.linearInterpolation(heatContent,pressureLevels,levels=newDepth)
    # Fix bounds
    newDepth = heatContent_depthInterp.getAxis(0)
    newDepth.setBounds(newDepth_bounds)
    del(newDepth_bounds)
    newDepth.id             = 'depth2'
    newDepth.units_long     = 'decibar (pressure)'
    newDepth.positive       = 'down'
    newDepth.long_name      = 'sea_water_pressure'
    newDepth.standard_name  = 'sea_water_pressure'
    newDepth.units          = 'decibar'
    newDepth.axis           = 'Z'
    
    #print 'hc_interp:',heatContent_depthInterp.min(),heatContent_depthInterp.max()

    # Integrate to 700 dbar - inputs heatContent
    heatContent_depthInteg = cdu.averager(heatContent_depthInterp[0:14,...],axis=0,weights='weighted',action='sum')(squeeze=1) # Calculate depth-weighted-integrated thetao
    #print heatContent_depthInteg.shape

    # Interpolate in x,y - inputs heatContent
    tmp1 = heatContent_depthInteg.regrid(mask.getGrid(),regridTool='esmf',regridMethod='linear') ; # Use defaults - ,coordSys='deg',diag = {},periodicity=1)
    #print tmp1.shape
    
    tmp1 = mv.where(tmp1<0,0,tmp1) ; # Fix for negative values
    
    # Infill - inputs heatContent
    # Create inputs for interpolation
    points = np.zeros([(mask.shape[0]*mask.shape[1]),2]) ; # Create 25380 vectors of lon/lat
    latcounter = 0 ; loncounter = 0
    for count,data in enumerate(points):
        if not np.mod(count,180) and not count == 0:
            latcounter = latcounter + 1
            loncounter = 0
        points[count,0] = mask.getLatitude().getValue()[latcounter]
        points[count,1] = mask.getLongitude().getValue()[loncounter]
        loncounter = loncounter + 1
    del(count,data,latcounter,loncounter); gc.collect()
    valid = np.logical_not(tmp1.mask) ; # Get inverted-logic boolean mask from variable
    if valid.size == 1:
        print '** No valid mask found, skipping **'
        return
    valid = valid.flatten() ; # Flatten 2D to 1D

    #maskFilled  = mask(tmp,points,valid)
    interpolant = interpolate.LinearNDInterpolator(points[valid,:],np.array(tmp1.flatten())[valid]) ; # Create interpolant
    maskFill    = interpolant(points[:,0].squeeze(),points[:,1].squeeze()) ; # Use interpolant to create filled matrix
    maskFill    = np.reshape(maskFill,mask.shape) ; # Resize to original dimensions    
    
    # Fix issues with interpolant
    tmp2 = mv.where(np.isnan(maskFill),1e+20,maskFill) ; # Fix for NaN values
    tmp2 = mv.where(tmp2>tmp1.max(),0,tmp2) ; # Fix for max values
    tmp2 = mv.where(tmp2<tmp1.min(),0,tmp2) ; # Fix for min values
    tmp = mv.masked_where(mask.mask,tmp2)
    #print tmp.shape
        
    # Redress variable
    heatContent                 = cdm.createVariable([tmp],id='heatContent')
    depthInt                    = cdm.createAxis([350],id='depth')
    depthInt.setBounds(np.array([0,700]))
    depthInt.units_long     = 'decibar (pressure)'
    depthInt.positive       = 'down'
    depthInt.long_name      = 'sea_water_pressure'
    depthInt.standard_name  = 'sea_water_pressure'
    depthInt.units          = 'decibar'
    depthInt.axis           = 'Z'
    heatContent.setAxis(0,depthInt)
    heatContent.setAxis(1,mask.getAxis(0))
    heatContent.setAxis(2,mask.getAxis(1))
    heatContent.units_long      = 'Joules'
    heatContent.long_name       = 'sea_water_heat_content'
    heatContent.standard_name   = 'sea_water_heat_content'
    heatContent.units           = 'J'
    
    return heatContent ; #,tmp1 ; # 0-700 dbar standard masked variable
