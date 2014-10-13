# -*- coding: utf-8 -*-
import gc,os#,sys
import cdms2 as cdm
import cdutil as cdu
import MV2 as mv
import numpy as np
np.seterr(all='ignore') ; # Cautious use of this turning all error reporting off - shouldn't be an issue as using masked arrays
import seawater as sw ; # was seawater.csiro
from durolib import getGitInfo,globalAttWrite
from numpy import array,isnan,tile,shape,transpose ; #mod
from string import replace
#import matplotlib as plt
#from matplotlib.cm import RdBu_r
#import seawater.gibbs as teos10

# Set netcdf file criterion - turned on from default 0s
cdm.setCompressionWarnings(0) ; # Suppress warnings
cdm.setNetcdfShuffleFlag(0)
cdm.setNetcdfDeflateFlag(1)
cdm.setNetcdfDeflateLevelFlag(9)
# Hi compression: 1.4Gb file ; # Single salt variable
# No compression: 5.6Gb ; Standard (compression/shuffling): 1.5Gb ; Hi compression w/ shuffling: 1.5Gb
cdm.setAutoBounds(1) ; # Ensure bounds on time and depth axes are generated


def scrubNaNAndMask(var,maskVar):
    # Check for NaNs
    nanvals = isnan(var)
    var[nanvals] = 1e+20
    var = mv.masked_where(maskVar>=1e+20,var)
    var = mv.masked_where(maskVar.mask,var)
    return var


def makeSteric(salinity,salinityChg,temp,tempChg,outFileName,thetao,pressure):
    """
    The makeSteric() function takes 3D (not temporal) arguments and creates
    heat content and steric fields which are written to a specified outfile
    
    Author: Paul J. Durack : pauldurack@llnl.gov : @durack1.
    Created on Thu Jul 18 13:03:37 2013.
    
    Inputs:
    ------
    - salinity(lev,lat,lon) - 3D array for the climatological period.
    - salinityChg(lev,lat,lon) - 3D array for the temporal change period.
    - temp(lev,lat,lon) - 3D array for the climatological period either in-situ or potential temperature.
    - tempChg(lev,lat,lon) - 3D array for the temporal change period as with temp, either in-situ or potential temperature.
    - outFileName(str) - output filename with full path specified.
    - thetao(bool) - boolean value specifying either in-situ or potential temperature arrays provided.
    - pressure(bool) - boolean value specifying whether lev-coordinate is pressure (dbar) or depth (m).
    
    Usage:
    ------
        >>> from makeStericLib import makeSteric
        >>> makeSteric(salinity,salinityChg,thetao,thetaoChg,'outfile.nc',True,False)
    
    Notes:
    -----
    - PJD 18 Jul 2013 - Validated Ishii v6.13 data against WOA94 - checks out ok. Units: dyn decimeter compared to http://www.nodc.noaa.gov/OC5/WOA94/dyn.html uses cm (not decimeter; x 10) 
    - PJD 18 Jul 2013 - Added attribute scrub to incoming variables (so,so_chg,temp,temp_chg) to maintain output consistency
    - PJD 22 Jul 2013 - Added name attributes to so and temp variables, added units to so_chg
    - PJD 22 Jul 2013 - removed duplicated code by converting repetition to function scrubNaNAndMask
    - PJD 23 Jul 2013 - Further cleaned up so,so_chg,temp,temp_chg outputs specifying id/name attributes
    - PJD  5 Aug 2013 - Updated python-seawater library to version 3.3.1 from github repo, git clone http://github.com/ocefpaf/python-seawater, python setup.py install --user
    - PJD  7 Aug 2013 - FIXED: thetao rather than in-situ temperature propagating throughout calculations
    - PJD  7 Aug 2013 - Replaced looping with 3D gpan
    - PJD  7 Aug 2013 - Further code duplication cleanup
    - PJD  8 Aug 2013 - FIXED: scrubNanAndMask function type/mask/grid issue - encase sw arguments in np.array() (attempt to strip cdms fluff)
    - PJD  8 Aug 2013 - FIXED: removed depth variable unit edits - not all inputs are depth (m)
    - PJD 15 Aug 2013 - Increased interpolated field resolution [200,300,500,700,1000,1500,1800,2000] - [5,10,20,30,40,50,75,100,125,150,200, ...]
    - PJD 18 Aug 2013 - AR5 hard coded rho=1020,cp=4187 == 4.3e6 vs Ishii 1970 rho.mean=1024,cp.mean=3922 == 4.1e6 ~5% too high
    - PJD 13 Jan 2014 - Corrected steric_height_anom and steric_height_thermo_anom to true anomaly fields, needed to remove climatology
    - PJD  3 May 2014 - Turned off thetao conversion, although convert to numpy array rather than cdms2 transient variable
    - PJD 13 Oct 2014 - Added seawater_library_version as a global attribute
    - PJD 13 Oct 2014 - FIXED: bug with calculation of rho_halo variable was calculating gpan
    - PJD 13 Oct 2014 - Added alternate calculation of halosteric anomaly (direct salinity anomaly calculation, rather than total-thermosteric)
    - PJD 13 Oct 2014 - Added makeSteric_version as a global attribute
    - TODO: Better deal with insitu vs thetao variables
    - TODO: Query Charles on why *.name attributes are propagating
    - TODO: validate outputs and compare to matlab versions - 10e-7 errors.
    """

    # Remap all variables to short names
    so          = salinity
    so_chg      = salinityChg
    temp        = temp
    temp_chg    = tempChg
    del(salinity,salinityChg,tempChg) ; gc.collect()

    # Strip attributes to maintain consistency between datasets
    for count,x in enumerate(so.attributes.keys()):
        delattr(so,x)
    #print so.listattributes() ; # Print remaining attributes
    for count,x in enumerate(so_chg.attributes.keys()):
        delattr(so_chg,x)
    for count,x in enumerate(temp.attributes.keys()):
        delattr(temp,x)
    for count,x in enumerate(temp_chg.attributes.keys()):
        delattr(temp_chg,x)
    del(count,x)
    
    # Create z-coordinate from salinity input
    if not pressure:
        z_coord                         = so.getAxis(0)
        y_coord                         = so.getAxis(1)
        y_coord                         = tile(y_coord,(so.shape[2],1)).transpose()
        depth_levels                    = tile(z_coord.getValue(),(so.shape[2],so.shape[1],1)).transpose()
        pressure_levels                 = sw.pres(np.array(depth_levels),np.array(y_coord))
        del(z_coord,y_coord,depth_levels) ; gc.collect()
    else:
        pressure_levels                 = so.getAxis(0)
        pressure_levels                 = transpose(tile(pressure_levels,(so.shape[2],so.shape[1],1)))
    
    pressure_levels                 = cdm.createVariable(pressure_levels,id='pressure_levels')
    pressure_levels.setAxis(0,so.getAxis(0))
    pressure_levels.setAxis(1,so.getAxis(1))
    pressure_levels.setAxis(2,so.getAxis(2))
    pressure_levels.id              = 'pressure_levels'
    pressure_levels.units_long      = 'decibar (pressure)'
    pressure_levels.positive        = 'down'
    pressure_levels.long_name       = 'sea_water_pressure'
    pressure_levels.standard_name   = 'sea_water_pressure'
    pressure_levels.units           = 'decibar'
    pressure_levels.axis            = 'Z'
    
    # Cleanup depth axis attributes
    depth               = so.getAxis(0)
    depth.id            = 'depth'
    depth.name          = 'depth'
    depth.long_name     = 'depth'
    depth.standard_name = 'depth'
    depth.axis          = 'Z'
    so.setAxis(0,depth)
    so_chg.setAxis(0,depth)
    temp.setAxis(0,depth)
    temp_chg.setAxis(0,depth)
    del(depth)
    
    # Convert using python-seawater library (v3.3.1 - 130807)
    if thetao:
        # Process potential temperature to in-situ - default conversion sets reference pressure to 0 (surface)
        #temp_chg                = sw.temp(np.array(so),np.array(temp_chg),np.array(pressure_levels)); # units degrees C
        #temp                    = sw.temp(np.array(so),np.array(temp),np.array(pressure_levels)); # units degrees C
        #temp_chg                = sw.ptmp(np.array(so),np.array(temp_chg),np.array(pressure_levels),np.array(pressure_levels)); # units degrees C
        #temp                    = sw.ptmp(np.array(so),np.array(temp),np.array(pressure_levels),np.array(pressure_levels)); # units degrees C
        temp_chg                = np.array(temp_chg); # units degrees C
        temp                    = np.array(temp); # units degrees C
    
    # Climatologies - rho,cp,steric_height
    rho                         = sw.dens(np.array(so),np.array(temp),np.array(pressure_levels)) ; # units kg m-3
    cp                          = sw.cp(np.array(so),np.array(temp),np.array(pressure_levels)) ; # units J kg-1 C-1
    steric_height               = sw.gpan(np.array(so),np.array(temp),np.array(pressure_levels)) ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    
    # Halosteric - rho,cp
    ss                          = map(array,(so+so_chg))
    rho_halo                    = sw.dens(np.array(ss),np.array(temp),np.array(pressure_levels)) ; # units kg m-3
    cp_halo                     = sw.cp(np.array(ss),np.array(temp),np.array(pressure_levels)) ; # units J kg-1 C-1
    tmp                         = sw.gpan(np.array(ss),np.array(temp),np.array(pressure_levels)) ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    steric_height_halo_anom2    = tmp-steric_height ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    
    # Full steric - steric_height
    tt                          = map(array,(temp+temp_chg))
    tmp                         = sw.gpan(np.array(ss),np.array(tt),np.array(pressure_levels)) ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    steric_height_anom          = tmp-steric_height ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    del(ss,tmp) ; gc.collect()
    
    # Thermosteric - rho,cp,steric_height
    rho_thermo                  = sw.dens(np.array(so),np.array(tt),np.array(pressure_levels)) ; # units kg m-3 
    cp_thermo                   = sw.cp(np.array(so),np.array(tt),np.array(pressure_levels)) ; # units J kg-1 C-1
    tmp                         = sw.gpan(np.array(so),np.array(tt),np.array(pressure_levels)) ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    steric_height_thermo_anom   = tmp-steric_height ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)
    del(tt,tmp) ; gc.collect()    

    # Halosteric - steric_height
    steric_height_halo_anom     = steric_height_anom-steric_height_thermo_anom ; # units m3 kg-1 Pa == m2 s-2 == J kg-1 (dynamic decimeter)

    # Create heat content
    heat_content                = np.array(temp)*np.array(rho)*np.array(cp) ; # units J
    heat_content_sanom          = np.array(temp)*np.array(rho_halo)*np.array(cp_halo) ; # units J
    heat_content_tanom          = np.array(temp_chg)*np.array(rho)*np.array(cp) ; # units J
    #heat_content_tanom          = np.array(temp_chg)*np.array(1020)*np.array(4187) ; # units J - try hard-coded - AR5 numbers
    heat_content_tsanom         = np.array(temp_chg)*np.array(rho_halo)*np.array(cp_halo) ; # units J
    
    # Correct all instances of NaN values and fix masks - applied before cdms variables are created otherwise names/ids/attributes are reset
    temp                        = scrubNaNAndMask(temp,so)
    temp_chg                    = scrubNaNAndMask(temp_chg,so)
    rho                         = scrubNaNAndMask(rho,so)
    cp                          = scrubNaNAndMask(cp,so)
    rho_halo                    = scrubNaNAndMask(rho_halo,so)
    cp_halo                     = scrubNaNAndMask(cp_halo,so)
    rho_thermo                  = scrubNaNAndMask(rho_thermo,so)
    cp_thermo                   = scrubNaNAndMask(cp_thermo,so)
    steric_height               = scrubNaNAndMask(steric_height,so)
    steric_height_anom          = scrubNaNAndMask(steric_height_anom,so)
    steric_height_thermo_anom   = scrubNaNAndMask(steric_height_thermo_anom,so)
    steric_height_halo_anom     = scrubNaNAndMask(steric_height_halo_anom,so)
    steric_height_halo_anom2    = scrubNaNAndMask(steric_height_halo_anom2,so)
    heat_content                = scrubNaNAndMask(heat_content,so)
    heat_content_sanom          = scrubNaNAndMask(heat_content_sanom,so)
    heat_content_tanom          = scrubNaNAndMask(heat_content_tanom,so)
    heat_content_tsanom         = scrubNaNAndMask(heat_content_tsanom,so)
    
    # Recreate and redress variables
    so.id                           = 'so_mean'
    so.units                        = '1e-3'
    so_chg.id                       = 'so_chg'
    so_chg.units                    = '1e-3'
    temp                            = cdm.createVariable(temp,id='temp_mean')
    temp.setAxis(0,so.getAxis(0))
    temp.setAxis(1,so.getAxis(1))
    temp.setAxis(2,so.getAxis(2))
    temp.units                      = 'degrees_C'
    temp_chg                        = cdm.createVariable(temp_chg,id='temp_chg')
    temp_chg.setAxis(0,so.getAxis(0))
    temp_chg.setAxis(1,so.getAxis(1))
    temp_chg.setAxis(2,so.getAxis(2))   
    temp_chg.units                  = 'degrees_C'    
    rho                             = cdm.createVariable(rho,id='rho')
    rho.setAxis(0,so.getAxis(0))
    rho.setAxis(1,so.getAxis(1))
    rho.setAxis(2,so.getAxis(2))
    rho.name                        = 'density_mean'
    rho.units                       = 'kg m^-3'
    cp                              = cdm.createVariable(cp,id='cp')
    cp.setAxis(0,so.getAxis(0))
    cp.setAxis(1,so.getAxis(1))
    cp.setAxis(2,so.getAxis(2))
    cp.name                         = 'heat_capacity_mean'
    cp.units                        = 'J kg^-1 C^-1'
    rho_halo                        = cdm.createVariable(rho_halo,id='rho_halo')
    rho_halo.setAxis(0,so.getAxis(0))
    rho_halo.setAxis(1,so.getAxis(1))
    rho_halo.setAxis(2,so.getAxis(2))
    rho_halo.name                   = 'density_mean_halo'
    rho_halo.units                  = 'kg m^-3'
    cp_halo                         = cdm.createVariable(cp_halo,id='cp_halo')
    cp_halo.setAxis(0,so.getAxis(0))
    cp_halo.setAxis(1,so.getAxis(1))
    cp_halo.setAxis(2,so.getAxis(2))
    cp_halo.name                    = 'heat_capacity_mean_halo'
    cp_halo.units                   = 'J kg^-1 C^-1'
    rho_thermo                      = cdm.createVariable(rho_thermo,id='rho_thermo')
    rho_thermo.setAxis(0,so.getAxis(0))
    rho_thermo.setAxis(1,so.getAxis(1))
    rho_thermo.setAxis(2,so.getAxis(2))
    rho_thermo.name                 = 'density_mean_thermo'
    rho_thermo.units                = 'kg m^-3'
    cp_thermo                       = cdm.createVariable(cp_thermo,id='cp_thermo')
    cp_thermo.setAxis(0,so.getAxis(0))
    cp_thermo.setAxis(1,so.getAxis(1))
    cp_thermo.setAxis(2,so.getAxis(2))
    cp_thermo.name                  = 'heat_capacity_mean_thermo'
    cp_thermo.units                 = 'J kg^-1 C^-1'
    steric_height                   = cdm.createVariable(steric_height,id='steric_height')
    steric_height.setAxis(0,so.getAxis(0))
    steric_height.setAxis(1,so.getAxis(1))
    steric_height.setAxis(2,so.getAxis(2))
    steric_height.units             = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_anom              = cdm.createVariable(steric_height_anom,id='steric_height_anom')
    steric_height_anom.setAxis(0,so.getAxis(0))
    steric_height_anom.setAxis(1,so.getAxis(1))
    steric_height_anom.setAxis(2,so.getAxis(2))
    steric_height_anom.units        = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_thermo_anom       = cdm.createVariable(steric_height_thermo_anom,id='steric_height_thermo_anom')
    steric_height_thermo_anom.setAxis(0,so.getAxis(0))
    steric_height_thermo_anom.setAxis(1,so.getAxis(1))
    steric_height_thermo_anom.setAxis(2,so.getAxis(2))
    steric_height_thermo_anom.units = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_halo_anom         = cdm.createVariable(steric_height_halo_anom,id='steric_height_halo_anom')
    steric_height_halo_anom.setAxis(0,so.getAxis(0))
    steric_height_halo_anom.setAxis(1,so.getAxis(1))
    steric_height_halo_anom.setAxis(2,so.getAxis(2))
    steric_height_halo_anom.units   = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_halo_anom2         = cdm.createVariable(steric_height_halo_anom2,id='steric_height_halo_anom2')
    steric_height_halo_anom2.setAxis(0,so.getAxis(0))
    steric_height_halo_anom2.setAxis(1,so.getAxis(1))
    steric_height_halo_anom2.setAxis(2,so.getAxis(2))
    steric_height_halo_anom2.units   = 'm^3 kg^-1 Pa (dynamic decimeter)'
    heat_content                    = cdm.createVariable(heat_content,id='heat_content')
    heat_content.setAxis(0,so.getAxis(0))
    heat_content.setAxis(1,so.getAxis(1))
    heat_content.setAxis(2,so.getAxis(2))
    heat_content.units         = 'J'
    heat_content_sanom              = cdm.createVariable(heat_content_sanom,id='heat_content_sanom')
    heat_content_sanom.setAxis(0,so.getAxis(0))
    heat_content_sanom.setAxis(1,so.getAxis(1))
    heat_content_sanom.setAxis(2,so.getAxis(2))
    heat_content_sanom.units        = 'J'
    heat_content_tanom              = cdm.createVariable(heat_content_tanom,id='heat_content_tanom')
    heat_content_tanom.setAxis(0,so.getAxis(0))
    heat_content_tanom.setAxis(1,so.getAxis(1))
    heat_content_tanom.setAxis(2,so.getAxis(2))
    heat_content_tanom.units        = 'J'
    heat_content_tsanom             = cdm.createVariable(heat_content_tsanom,id='heat_content_tsanom')
    heat_content_tsanom.setAxis(0,so.getAxis(0))
    heat_content_tsanom.setAxis(1,so.getAxis(1))
    heat_content_tsanom.setAxis(2,so.getAxis(2))
    heat_content_tsanom.units       = 'J'
    
    # Create model-based depth index for subset target levels
    newdepth = np.array([5,10,20,30,40,50,75,100,125,150,200,300,500,700,1000,1500,1800,2000]).astype('f');
    newdepth_bounds = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,75],[75,100],[100,125],[125,150],
    [150,200],[200,300],[300,500],[500,700],[700,1000],[1000,1500],[1500,1800],[1800,2000]]).astype('f')
    #newdepth = np.array([200,300,500,700,1000,1500,1800,2000]).astype('f');
    #newdepth_bounds = np.array([[0,200],[200,300],[300,500],[500,700],[700,1000],[1000,1500],[1500,1800],[1800,2000]]).astype('f')
    
    # Interpolate to depths
    so_depthInterp                          = cdu.linearInterpolation(so,pressure_levels,levels=newdepth)
    temp_depthInterp                        = cdu.linearInterpolation(temp,pressure_levels,levels=newdepth)
    steric_height_depthInterp               = cdu.linearInterpolation(steric_height,pressure_levels,levels=newdepth)
    steric_height_anom_depthInterp          = cdu.linearInterpolation(steric_height_anom,pressure_levels,levels=newdepth)
    steric_height_thermo_anom_depthInterp   = cdu.linearInterpolation(steric_height_thermo_anom,pressure_levels,levels=newdepth)
    steric_height_halo_anom_depthInterp     = cdu.linearInterpolation(steric_height_halo_anom,pressure_levels,levels=newdepth)
    steric_height_halo_anom2_depthInterp    = cdu.linearInterpolation(steric_height_halo_anom2,pressure_levels,levels=newdepth)
    heat_content_sanom_depthInterp          = cdu.linearInterpolation(heat_content_sanom,pressure_levels,levels=newdepth)
    heat_content_tanom_depthInterp          = cdu.linearInterpolation(heat_content_tanom,pressure_levels,levels=newdepth)
    heat_content_tsanom_depthInterp         = cdu.linearInterpolation(heat_content_tanom,pressure_levels,levels=newdepth)
    
    # Fix masks - applied before cdms variables are created otherwise names/ids/attributes are reset
    temp_depthInterp                        = scrubNaNAndMask(temp_depthInterp,so_depthInterp)
    steric_height_depthInterp               = scrubNaNAndMask(steric_height_depthInterp,so_depthInterp)
    steric_height_anom_depthInterp          = scrubNaNAndMask(steric_height_anom_depthInterp,so_depthInterp)
    steric_height_thermo_anom_depthInterp   = scrubNaNAndMask(steric_height_thermo_anom_depthInterp,so_depthInterp)
    steric_height_halo_anom_depthInterp     = scrubNaNAndMask(steric_height_halo_anom_depthInterp,so_depthInterp)
    steric_height_halo_anom2_depthInterp    = scrubNaNAndMask(steric_height_halo_anom2_depthInterp,so_depthInterp)
    heat_content_sanom_depthInterp          = scrubNaNAndMask(heat_content_sanom_depthInterp,so_depthInterp)
    heat_content_tanom_depthInterp          = scrubNaNAndMask(heat_content_tanom_depthInterp,so_depthInterp)
    heat_content_tsanom_depthInterp         = scrubNaNAndMask(heat_content_tsanom_depthInterp,so_depthInterp)
    
    # Fix bounds
    newdepth = so_depthInterp.getAxis(0)
    newdepth.setBounds(newdepth_bounds)
    del(newdepth_bounds)
    newdepth.id             = 'depth2'
    newdepth.units_long     = 'decibar (pressure)'
    newdepth.positive       = 'down'
    newdepth.long_name      = 'sea_water_pressure'
    newdepth.standard_name  = 'sea_water_pressure'
    newdepth.units          = 'decibar'
    newdepth.axis           = 'Z'
    
    # Assign corrected bounds
    so_depthInterp.setAxis(0,newdepth)
    temp_depthInterp.setAxis(0,newdepth)
    steric_height_depthInterp.setAxis(0,newdepth)
    steric_height_anom_depthInterp.setAxis(0,newdepth)
    steric_height_thermo_anom_depthInterp.setAxis(0,newdepth)
    steric_height_halo_anom_depthInterp.setAxis(0,newdepth)
    steric_height_halo_anom2_depthInterp.setAxis(0,newdepth)
    heat_content_sanom_depthInterp.setAxis(0,newdepth)
    heat_content_tanom_depthInterp.setAxis(0,newdepth)
    heat_content_tsanom_depthInterp.setAxis(0,newdepth)
    
    # Average/integrate to surface - configure bounds
    # Preallocate arrays
    so_depthAve                     = np.ma.zeros([len(newdepth),shape(so)[1],shape(so)[2]])
    temp_depthAve                   = so_depthAve.copy()
    heat_content_sanom_depthInteg   = so_depthAve.copy()
    heat_content_tanom_depthInteg   = so_depthAve.copy()
    heat_content_tsanom_depthInteg  = so_depthAve.copy()
    for count,depth in enumerate(newdepth):
        tmp = cdu.averager(so_depthInterp[0:(count+1),...],axis=0,weights='weighted',action='average')
        so_depthAve[count,]                     = tmp;
        tmp = cdu.averager(temp_depthInterp[0:(count+1),...],axis=0,weights='weighted',action='average')
        temp_depthAve[count,]                   = tmp;
        tmp = cdu.averager(heat_content_sanom_depthInterp[0:(count+1),...],axis=0,weights='weighted',action='sum')
        heat_content_sanom_depthInteg[count,]   = tmp
        tmp = cdu.averager(heat_content_tanom_depthInterp[0:(count+1),...],axis=0,weights='weighted',action='sum')
        heat_content_tanom_depthInteg[count,]   = tmp
        tmp = cdu.averager(heat_content_tsanom_depthInterp[0:(count+1),...],axis=0,weights='weighted',action='sum')
        heat_content_tsanom_depthInteg[count,]  = tmp
    del(heat_content_tanom_depthInterp,heat_content_tsanom_depthInterp); gc.collect()
    
    # Fix masks - applied before cdms variables are created otherwise names/ids/attributes are reset
    so_depthAve = scrubNaNAndMask(so_depthAve,so_depthInterp)
    temp_depthAve = scrubNaNAndMask(temp_depthAve,so_depthInterp)
    heat_content_sanom_depthInteg = scrubNaNAndMask(heat_content_sanom_depthInteg,so_depthInterp)
    heat_content_tanom_depthInteg = scrubNaNAndMask(heat_content_tanom_depthInteg,so_depthInterp)
    heat_content_tsanom_depthInteg = scrubNaNAndMask(heat_content_tsanom_depthInteg,so_depthInterp)
    del(so_depthInterp)
    
    # Convert numpy arrays to cdms objects
    heat_content_sanom_depthInteg               = cdm.createVariable(heat_content_sanom_depthInteg,id='heat_content_sanom_depthInteg')
    heat_content_sanom_depthInteg.id            = 'heat_content_sanom_depthInteg'
    heat_content_sanom_depthInteg.setAxis(0,newdepth)
    heat_content_sanom_depthInteg.setAxis(1,so.getAxis(1))
    heat_content_sanom_depthInteg.setAxis(2,so.getAxis(2))
    heat_content_sanom_depthInteg.units         = 'J'
    heat_content_tanom_depthInteg               = cdm.createVariable(heat_content_tanom_depthInteg,id='heat_content_tanom_depthInteg')
    heat_content_tanom_depthInteg.id            = 'heat_content_tanom_depthInteg'
    heat_content_tanom_depthInteg.setAxis(0,newdepth)
    heat_content_tanom_depthInteg.setAxis(1,so.getAxis(1))
    heat_content_tanom_depthInteg.setAxis(2,so.getAxis(2))
    heat_content_tanom_depthInteg.units         = 'J'
    heat_content_tsanom_depthInteg              = cdm.createVariable(heat_content_tsanom_depthInteg,id='heat_content_tsanom_depthInteg')
    heat_content_tsanom_depthInteg.id           = 'heat_content_tsanom_depthInteg'
    heat_content_tsanom_depthInteg.setAxis(0,newdepth)
    heat_content_tsanom_depthInteg.setAxis(1,so.getAxis(1))
    heat_content_tsanom_depthInteg.setAxis(2,so.getAxis(2))
    heat_content_tsanom_depthInteg.units        = 'J'
    so_depthAve                                 = cdm.createVariable(so_depthAve,id='so_depthAve')
    so_depthAve.id                              = 'so_depthAve'
    so_depthAve.setAxis(0,newdepth)
    so_depthAve.setAxis(1,so.getAxis(1))
    so_depthAve.setAxis(2,so.getAxis(2))
    so_depthAve.units                           = '1e-3'
    temp_depthAve                               = cdm.createVariable(temp_depthAve,id='temp_depthAve')
    temp_depthAve.id                            = 'temp_depthAve'
    temp_depthAve.setAxis(0,newdepth)
    temp_depthAve.setAxis(1,so.getAxis(1))
    temp_depthAve.setAxis(2,so.getAxis(2))
    temp_depthAve.units                         = 'degrees_C'
    steric_height_depthInterp                   = cdm.createVariable(steric_height_depthInterp,id='steric_height_depthInterp')
    steric_height_depthInterp.setAxis(0,newdepth)
    steric_height_depthInterp.setAxis(1,so.getAxis(1))
    steric_height_depthInterp.setAxis(2,so.getAxis(2))
    steric_height_depthInterp.units             = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_anom_depthInterp              = cdm.createVariable(steric_height_anom_depthInterp,id='steric_height_anom_depthInterp')
    steric_height_anom_depthInterp.setAxis(0,newdepth)
    steric_height_anom_depthInterp.setAxis(1,so.getAxis(1))
    steric_height_anom_depthInterp.setAxis(2,rho.getAxis(2))
    steric_height_anom_depthInterp.units        = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_thermo_anom_depthInterp       = cdm.createVariable(steric_height_thermo_anom_depthInterp,id='steric_height_thermo_anom_depthInterp')
    steric_height_thermo_anom_depthInterp.setAxis(0,newdepth)
    steric_height_thermo_anom_depthInterp.setAxis(1,so.getAxis(1))
    steric_height_thermo_anom_depthInterp.setAxis(2,so.getAxis(2))
    steric_height_thermo_anom_depthInterp.units = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_halo_anom_depthInterp         = cdm.createVariable(steric_height_halo_anom_depthInterp,id='steric_height_halo_anom_depthInterp')
    steric_height_halo_anom_depthInterp.setAxis(0,newdepth)
    steric_height_halo_anom_depthInterp.setAxis(1,so.getAxis(1))
    steric_height_halo_anom_depthInterp.setAxis(2,so.getAxis(2))
    steric_height_halo_anom_depthInterp.units   = 'm^3 kg^-1 Pa (dynamic decimeter)'
    steric_height_halo_anom2_depthInterp         = cdm.createVariable(steric_height_halo_anom2_depthInterp,id='steric_height_halo_anom2_depthInterp')
    steric_height_halo_anom2_depthInterp.setAxis(0,newdepth)
    steric_height_halo_anom2_depthInterp.setAxis(1,so.getAxis(1))
    steric_height_halo_anom2_depthInterp.setAxis(2,so.getAxis(2))
    steric_height_halo_anom2_depthInterp.units   = 'm^3 kg^-1 Pa (dynamic decimeter)'
    # Cleanup workspace
    del(newdepth) ; gc.collect()
    
    
    # Write variables to file
    if os.path.isfile(outFileName):
        os.remove(outFileName)
    filehandle = cdm.open(outFileName,'w')
    # Global attributes
    globalAttWrite(filehandle,options=None) ; # Use function to write standard global atts
    # Write seawater version
    filehandle.seawater_library_version = sw.__version__
    # Write makeSteric version
    makeStericPath = str(makeSteric.__code__).split(' ')[6]
    makeStericPath = replace(replace(makeStericPath,'"',''),',','') ; # Clean scraped path
    filehandle.makeSteric_version = ' '.join(getGitInfo(makeStericPath)[0:3])
    # Master variables
    filehandle.write(so.astype('float32'))
    filehandle.write(so_chg.astype('float32'))
    filehandle.write(so_depthAve.astype('float32'))
    filehandle.write(temp.astype('float32'))
    filehandle.write(temp_chg.astype('float32'))
    filehandle.write(temp_depthAve.astype('float32'))
    # Derived variables
    filehandle.write(cp.astype('float32'))
    filehandle.write(cp_halo.astype('float32'))
    filehandle.write(cp_thermo.astype('float32'))    
    filehandle.write(rho.astype('float32'))
    filehandle.write(rho_halo.astype('float32'))
    filehandle.write(rho_thermo.astype('float32'))
    filehandle.write(heat_content.astype('float32'))
    filehandle.write(heat_content_sanom.astype('float32'))
    filehandle.write(heat_content_sanom_depthInteg.astype('float32'))
    filehandle.write(heat_content_tanom.astype('float32'))
    filehandle.write(heat_content_tanom_depthInteg.astype('float32'))
    filehandle.write(heat_content_tsanom.astype('float32'))
    filehandle.write(heat_content_tsanom_depthInteg.astype('float32'))
    filehandle.write(steric_height.astype('float32'))
    filehandle.write(steric_height_depthInterp.astype('float32'))
    filehandle.write(steric_height_anom.astype('float32'))
    filehandle.write(steric_height_anom_depthInterp.astype('float32'))
    filehandle.write(steric_height_halo_anom.astype('float32'))
    filehandle.write(steric_height_halo_anom2.astype('float32'))
    filehandle.write(steric_height_halo_anom_depthInterp.astype('float32'))
    filehandle.write(steric_height_halo_anom2_depthInterp.astype('float32'))
    filehandle.write(steric_height_thermo_anom.astype('float32'))
    filehandle.write(steric_height_thermo_anom_depthInterp.astype('float32'))
    filehandle.close()
    # Cleanup workspace
    del(outFileName) ; gc.collect()

## Heat content - quick plots to check we're on track - all look good
'''
plt.close('all')
## Heat content
plt.figure(1)
plt.pcolormesh(heat_content_tanom[20,...],cmap='RdBu_r') #,vmax=5e6,vmin=-5e6)
plt.title('heat_content_tanom:20')
plt.colorbar()
plt.figure(2)
plt.pcolormesh(heat_content_clim[6,...],cmap='RdBu_r',vmax=5e9,vmin=-5e9)
plt.hold(True);
plt.contour(heat_content_clim[6,...],vmax=5e9,vmin=-5e9)
plt.title('heat_content:6')
plt.colorbar()
# Heat content map looks similar in spatial structure to thermo 0-1800m from matlab
plt.figure(3)
plt.pcolor(so_depth_int[6,...],cmap='RdBu_r',vmax=36,vmin=34)
plt.hold(True);
plt.contour(so_depth_int[6,...],[34.125,34.25,34.625,34.75,35.125,35.25,35.625,35.75],colors='k')
plt.hold(True)
plt.contour(so_depth_int[6,...],[34,34.5,35,35.5,36],colors='k',linewidth=2);
plt.title('so_depth_int:6')
plt.colorbar()
## Steric height
plt.close('all')
plt.figure(1)
plt.pcolormesh(clim_steric_height[20,...],cmap='RdBu_r') ; #,vmax=5e6,vmin=-5e6)
plt.title('clim_steric_height:20')
plt.colorbar()
plt.figure(2)
plt.pcolormesh(anom_thermosteric_height[20,...],cmap='RdBu_r') ; #,vmax=7.5e9,vmin=-3e9)
plt.title('anom_thermosteric_height:20')
plt.colorbar()
plt.figure(3)
plt.pcolormesh(anom_halosteric_height[20,...],cmap='RdBu_r') ; #,vmax=7.5e9,vmin=-3e9)
plt.title('anom_halosteric_height:20')
plt.colorbar()
'''

## Steric fields - quick plots to check we're on track
'''
halo_1800 = steric_height_anom_depthInt[6,...]-steric_height_thermo_anom_depthInt[6,...]
halo_1800 = (halo_1800-halo_1800.mean())*1 ; # Convert decimeters -> mm (dm 1 == mm 100)
close('all')
figure(1)
plt.pcolormesh(halo_1800,cmap='RdBu_r',vmax=4,vmin=-4)
plt.hold(True)
plt.contour(halo_1800,[-4,-3,-2,-1,1,2,3,4],colors='k')
plt.hold(True)
plt.contour(halo_1800,[0],colors='k',linewidth=2);
plt.title('halo_1800')
plt.colorbar()
thermo_1800 = (steric_height_thermo_anom_depthInt[6,...]-steric_height_thermo_anom_depthInt[6,...].mean())*1 ; # Convert decimeters -> mm
plt.figure(2)
plt.pcolormesh(thermo_1800,cmap='RdBu_r',vmax=4,vmin=-4)
hold(True)
plt.contour(thermo_1800,[-4,-3,-2,-1,1,2,3,4],colors='k')
plt.hold(True)
plt.contour(thermo_1800,[0],colors='k',linewidth=2);
plt.title('thermo_1800')
plt.colorbar()
'''
