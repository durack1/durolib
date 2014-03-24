# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:49:02 2013

Paul J. Durack 27th May 2013

PJD  2 Jun 2013     - Added clearall and outer_locals functions
PJD 13 Jun 2013     - Updated global_att_write to globalAttWrite
PJD 13 Jun 2013     - Added writeToLog function
PJD 26 Jun 2013     - Added fixInterpAxis function
PJD 18 Jul 2013     - Sphinx docs/syntax http://pythonhosted.org/an_example_pypi_project/sphinx.html
PJD 23 Jul 2013     - Added fixVarUnits function
PJD 25 Jul 2013     - Updated fixVarUnits function to print and log changes
PJD  5 Aug 2013     - Added fitPolynomial function following Pete G's code example
PJD  9 Aug 2013     - Added writePacked function
PJD  9 Aug 2013     - Added keyboard function
PJD 22 Aug 2013     - Added setTimeBoundsYearly() to fixInterpAxis
                    - TODO: Consider implementing multivariate polynomial regression:
                      https://github.com/mrocklin/multipolyfit

This library contains all functions written to replicate matlab functionality in python

@author: durack1
"""

## Import common modules ##
import cdat_info,code,datetime,gc,os,pytz,sys,inspect
import cdms2 as cdm
import cdutil as cdu
#import genutil as genu
#import matplotlib as plt
import MV2 as mv
import numpy as np
#import scipy as sp
from numpy.core.fromnumeric import shape
from socket import gethostname
from string import replace
# Consider modules listed in /work/durack1/Shared/130103_data_SteveGriffies/130523_mplib_tips/importNPB.py
##

## Specify UVCDAT specific stuff ##
# Turn off cdat ping reporting - Does this speed up Spyder?
cdat_info.ping = False
# Set netcdf file criterion - turned on from default 0s
cdm.setCompressionWarnings(0) ; # Suppress warnings
cdm.setNetcdfShuffleFlag(0)
cdm.setNetcdfDeflateFlag(1)
cdm.setNetcdfDeflateLevelFlag(9)
# Hi compression: 1.4Gb file ; # Single salt variable
# No compression: 5.6Gb ; Standard (compression/shuffling): 1.5Gb ; Hi compression w/ shuffling: 1.5Gb
cdm.setAutoBounds(1) ; # Ensure bounds on time and depth axes are generated
##

## Define useful functions ##

def clearAll():
    """
    Documentation for clearall():
    -------
    The clearall() function purges all variables in global namespace
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage:
    ------
        >>> from durolib import clearall
        >>> clearall()
    
    Notes:
    -----
        Currently not working ...
    """
    for uniquevariable in [variable for variable in globals().copy() if variable[0] != "_" and variable != 'clearall']:
        del globals()[uniquevariable]


def environment():
    return False


def fitPolynomial(var,time,polyOrder):
    """
    Documentation for fitPolynomial(var):
    -------
    The fitPolynomial(var,time,polyOrder) function returns a new variable which is the polyOrder
    estimate of the variable argument
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage:
    ------
        >>> from durolib import fitPolynomial
        >>> var_cubic = fitPolynomial(var,time,polyOrder=3)
    
    Notes:
    -----
    - PJD  5 Aug 2013 - Implemented following examples from Pete G.
    - TODO: only works on 2D arrays, improve to work on 3D
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    """
    if polyOrder > 3:
        print "".join(['** fitPolynomial Error: >cubic fits not supported **',])
        return
    varFitted = mv.multiply(var,0.) ; # Preallocate output array    
    coefs,residuals,rank,singularValues,rcond = np.polyfit(time,var,polyOrder,full=True)
    for timeIndex in range(len(time)):
        timeVal = time[timeIndex]
        if polyOrder == 1:
            varFitted[timeIndex] = (coefs[0]*timeVal + coefs[1])
        elif polyOrder == 2:
            varFitted[timeIndex] = (coefs[0]*(timeVal**2) + coefs[1]*timeVal + coefs[2])
        elif polyOrder == 3:
            varFitted[timeIndex] = (coefs[0]*(timeVal**3) + coefs[1]*(timeVal**2) + coefs[2]*timeVal + coefs[3])
    return varFitted
    

def fixInterpAxis(var):
    """
    Documentation for fixInterpAxis(var):
    -------
    The fixInterpAxis(var) function corrects temporal axis so that genutil.statistics.linearregression
    returns coefficients which are unscaled by the time axis
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage:
    ------
        >>> from durolib import fixInterpAxis
        >>> (slope),(slope_err) = linearregression(fixInterpAxis(var),error=1,nointercept=1)
    
    Notes:
    -----
        ...
    """
    tind = range(shape(var)[0]) ; # Assume time axis is dimension 0
    t = cdm.createAxis(tind,id='time')
    t.units = 'years since 0-01-01 0:0:0.0'
    t.calendar = var.getTime().calendar
    cdu.times.setTimeBoundsYearly(t) ; # Explicitly set time bounds to yearly
    var.setAxis(0,t)
    return var


def fixVarUnits(var,varName,report=False,logFile=None):
    """
    Documentation for fixVarUnits():
    -------
    The fixVarUnits() function corrects units of salinity and converts thetao from K to degrees_C
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage:
    ------
        >>> from durolib import fixVarUnits
        >>> [var,var_fixed] = fixVarUnits(var,'so',True,'logfile.txt')
    
    Notes:
    -----
        ...
    """
    var_fixed = False
    if varName in ['so','sos']:
        if var.max() < 1. and var.mean() < 1.:
            if report:
                print "".join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))])
            if logFile is not None:
                writeToLog(logFile,"".join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
            var_ = var*1000
            var_.id = var.id
            var_.name = var.id
            for k in var.attributes.keys():
                setattr(var_,k,var.attributes[k])
            var = var_
            var_fixed = True
            if report:
                print "".join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))])
            if logFile is not None:
                writeToLog(logFile,"".join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
    elif varName in 'thetao':
        if var.max() > 50. and var.mean() > 265.:
            if report:
                print "".join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))])
            if logFile is not None:
                writeToLog(logFile,"".join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
            var_ = var-273.15
            var_.id = var.id
            var_.name = var.id
            for k in var.attributes.keys():
                setattr(var_,k,var.attributes[k])
            var = var_
            var_fixed = True
            if report:
                print "".join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))])
            if logFile is not None:
                writeToLog(logFile,"".join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))

    return var,var_fixed


def globalAttWrite(file_handle,options):
    """
    Documentation for globalAttWrite():
    -------
    The globalAttWrite() function writes standard global_attributes to an
    open netcdf specified by file_handle
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Returns:
    -------
           Nothing.
    Usage: 
    ------
        >>> from durolib import globalAttWrite
        >>> globalAttWrite(file_handle)
    
    Where file_handle is a handle to an open, writeable netcdf file
    
    Optional Arguments:
    -------------------
    option=optionalArguments   
    Restrictions: option has to be a string
    Default : ...
    
    You can pass option='SOMETHING', ...
    
    Examples:
    ---------
        >>> from durolib import globalAttWrite
        >>> f = cdms2.open('data_file_name','w')
        >>> globalAttWrite(f)
        # Writes standard global attributes to the netcdf file specified by file_handle
            
    Notes:
    -----
        When ...
    """
    # Create timestamp, corrected to UTC for history
    local                       = pytz.timezone("America/Los_Angeles")
    time_now                    = datetime.datetime.now();
    local_time_now              = time_now.replace(tzinfo = local)
    utc_time_now                = local_time_now.astimezone(pytz.utc)
    time_format                 = utc_time_now.strftime("%d-%m-%Y %H:%M:%S %p")
    file_handle.institution     = "Program for Climate Model Diagnosis and Intercomparison (LLNL)"
    file_handle.data_contact    = "Paul J. Durack; pauldurack@llnl.gov; +1 925 422 5208"
    file_handle.history         = "".join(['File processed: ',time_format,' UTC; San Francisco, CA, USA'])
    file_handle.host            = "".join([gethostname(),'; UVCDAT version: ',".".join(["%s" % el for el in cdat_info.version()]),
                                           '; Python version: ',replace(replace(sys.version,'\n','; '),') ;',');')])

def inpaint(array,method):
    #/work/durack1/csiro/Backup/110808/Z_dur041_linux/bin/inpaint_nans/inpaint_nans.m
    return False


def keyboard(banner=None):
    """
    Documentation for keyboard():
    -------
    The keyboard() function mimics matlab's keyboard function allowing control
    sent to the keyboard within a running script
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Returns:
    -------
           Nothing.
    Usage: 
    ------
        >>> from durolib import keyboard
        >>> keyboard()
    
    Examples:
    ---------
        ...
            
    Notes:
    -----
        ...
    """    
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner,local=namespace)
    except SystemExit:
        return
        

def outerLocals(depth=0):
    return inspect.getouterframes(inspect.currentframe())[depth+1][0].f_locals


def smooth(array,method):
    #/apps/MATLAB/R2011b/toolbox/matlab/specgraph/smooth3.m
    #/apps/MATLAB/R2011b/toolbox/curvefit/curvefit/smooth.m
    return False


def spyderClean():
    """
    Documentation for spyder_clean():
    -------
    The spyder_clean() function purges variables initialised upon startup
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage: 
    ------
        >>> from durolib import spyder_clean
        >>> spyder_clean()
    
    Notes:
    -----
        Currently not working ...
    """
    local_vars = outerLocals()
    if 'e' in local_vars:
        print 'yep..'
        del(e,pi,sctypeNA,typeNA)
        gc.collect()


def writeToLog(logFilePath,textToWrite):
    """
    Documentation for writeToLog(logFilePath,textToWrite):
    -------
    The writeToLog() function writes specified text to a text log file
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage: 
    ------
        >>> from durolib import writeToLog
        >>> writeToLog(~/somefile.txt,'text to write to log file')
    
    Notes:
    -----
        Current version appends a new line after each call to the function.
        File will be created if it doesn't already exist, otherwise new text
        will be appended to an existing log file.
    """
    if os.path.isfile(logFilePath):
        logHandle = open(logFilePath,'a') ; # Open to append
    else:
        logHandle = open(logFilePath,'w') ; # Open to write     
    logHandle.write("".join([textToWrite,'\n']))
    logHandle.close()
    
    
def writePacked(var,fileObject='tmp.nc'):
    """
    Documentation for writePacked(var,fileObject):
    -------
    The writePacked() function generates a 16-bit (int16) cdms2 variable and
    writes this to a netcdf file
    
    Author: Paul J. Durack : pauldurack@llnl.gov
    
    Usage: 
    ------
        >>> from durolib import writePacked
        >>> writePacked(var,'16bitPacked.nc')
    
    Notes:
    -----
        TODO: clean up fileObject existence..
        TODO: deal with incredibly slow write-times
    """
    varMin              = var.min()
    varMax              = var.max()
    var.scale_factor    = np.float32((varMax-varMin)/(2**16)-1)
    var.add_offset      = np.float32(varMin+var.scale_factor*(2**15))
    if 'tmp.nc' in fileObject:
        fileObject = cdm.open(fileObject,'w')
    fileObject.write((var-var.add_offset)/var.scale_factor,dtype=np.int16)
    return

##