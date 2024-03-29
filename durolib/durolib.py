#!/bin/env python2
# -*- coding: utf-8 -*-
"""
Documentation for durolib:
-----
Created on Mon May 27 16:49:02 2013

Paul J. Durack 27th May 2013

|  PJD  2 Jun 2013  - Added clearall and outer_locals functions
|  PJD 13 Jun 2013  - Updated global_att_write to globalAttWrite
|  PJD 13 Jun 2013  - Added writeToLog function
|  PJD 26 Jun 2013  - Added fixInterpAxis function
|  PJD 18 Jul 2013  - Sphinx docs/syntax http://pythonhosted.org/an_example_pypi_project/sphinx.html
|  PJD 23 Jul 2013  - Added fixVarUnits function
|  PJD 25 Jul 2013  - Updated fixVarUnits function to print and log changes
|  PJD  5 Aug 2013  - Added fitPolynomial function following Pete G's code example
|  PJD  9 Aug 2013  - Added writePacked function
|  PJD  9 Aug 2013  - Added keyboard function
|  PJD 22 Aug 2013  - Added setTimeBoundsYearly() to fixInterpAxis
|  PJD  1 Apr 2014  - Added trimModelList
|  PJD 20 Aug 2014  - Added mkDirNoOSErr and sysCallTimeout functions
|  PJD 13 Oct 2014  - Added getGitInfo function
|  PJD 20 Feb 2015  - Added makeCalendar function
|  PJD 30 Apr 2015  - Fixed off by one issue with partial years in makeCalendar
|  PJD 16 Jun 2015  - Added 'noid' option in globalAttWrite
|  PJD  3 Nov 2015  - Added globAndTrim, matchAndTrimBlanks and truncateVerInfo functions
|  PJD  3 Nov 2015  - Moved UV-CDAT packages into a try block (cdat_info,cdms2,cdtime,MV2)
|  PJD 17 Nov 2015  - Added daysBetween function
|  PJD 19 Nov 2015  - Added scrubNaNAndMask function
|  PJD  3 Dec 2015  - Added santerTime function
|  PJD 25 May 2016  - Removed pytz as standard library import
|  PJD  7 Jul 2016  - Updated makeCalendar to correct off by 1 error
|  PJD 15 Aug 2016  - Added readJsonCreateDict function
|  PJD 25 Aug 2016  - Added 'years' calendarStep argument to makeCalendar
|  PJD 26 Aug 2016  - Add shebang
|  PJD 31 Aug 2016  - Updated getGitInfo to include tag info
|  PJD 28 Nov 2016  - Updated getGitInfo to deal with tag information
|  PJD  8 Mar 2017  - Updated globalAttWrite to deal with UV-CDAT version tags
|  PJD 19 Apr 2018  - Corrected UTC offset, now correct times are reported - https://github.com/durack1/durolib/issues/20 & stub42/pytz/issues/12
|  PJD 18 Jun 2018  - Added cmipBranchTime function https://github.com/durack1/durolib/issues/6
|  PJD  6 Sep 2018  - Corrected fixInterpAxis time units
|  PJD  6 Sep 2018  - Updated globalAttWrite to reference CDAT (not UVCDAT) version
|  PJD 10 Oct 2018  - Updated to deal with new CMIPLib paths and filenames (CMIP6/CMIP5/cmip5 data) - diagnostic testing included in function docs
|  PJD 12 Mar 2019  - Updated trimModelList to deal with cmip-dyn paths
|  PJD 12 Mar 2019  - Updated setup.py to deal with CMIP5BranchTime.json
|  PJD 13 Mar 2019  - Updated to deal with new conda/setuptools distrib install - durolib_egg_path
|  PJD 10 Oct 2019  - Update to use __future__ print (py3)
|  PJD 10 Oct 2019  - Updated following "invalid token" error see
|                     https://stackoverflow.com/questions/36386346/syntaxerror-invalid-token
|  PJD 10 Oct 2019  - Update from urllib2 to urllib.request
|  PJD 10 Oct 2019  - Update string.replace syntax calls
|  PJD  2 Feb 2020  - Reverted urllib2 -> urllib change
|  PJD 24 Jul 2020  - Updated readJsonCreateDict to take a second arg, the urlPrefix
|  PJD 29 Jul 2020  - Updated readJsonCreateDict to correct case when urlPrefix not supplied
|                   - TODO: Consider implementing multivariate polynomial regression:
|                     https://github.com/mrocklin/multipolyfit
|  PJD 11 Nov 2020  - Update for Py3 (getGitInfo, readJsonCreateDict)
|  PJD 20 Jun 2023  - Updated getGitInfo to test gitTagErr.strip() == b'str' vs 'str' for py3

This library contains all functions written to replicate matlab functionality in python

@author: durack1
"""

## Import future print
from __future__ import print_function
## Import common modules ##
# import pdb
import calendar, code, datetime, errno, glob, inspect, json, os, pkg_resources, \
       re, ssl, string, sys, time
try:
    from urllib2 import urlopen # py2
except ImportError:
    from urllib.request import urlopen # py3
#import matplotlib as plt
import numpy as np
import subprocess
#import scipy as sp
from numpy import isnan,shape
from socket import gethostname
# Consider modules listed in /work/durack1/Shared/130103_data_SteveGriffies/130523_mplib_tips/importNPB.py

# Determine if local file or conda install
#print 'sys.argv[0]:',os.path.realpath(sys.argv[0])
#print '__file__:',os.path.realpath(__file__)
if 'site-packages' in os.path.realpath(__file__):
    durolib_egg_path = pkg_resources.resource_filename(pkg_resources.Requirement.parse('durolib'),'share/durolib/data')
else:
    #print os.path.realpath(__file__).split('/')[:-2]
    #print os.path.join([os.path.realpath(__file__).split('/')[:-2],'data'])
    srcPath = os.path.realpath(__file__).split('/')[:-2]
    srcPath = os.path.join(*srcPath)
    durolib_egg_path = os.path.join('/',srcPath,'data')
#print durolib_egg_path

# Move UV-CDAT packages into try block
try:
    import cdat_info
    # Turn off cdat ping reporting - Does this speed up Spyder?
    cdat_info.ping = False
    import cdms2 as cdm
    import cdtime as cdt
    import cdutil as cdu
    #import genutil as genu
    import MV2 as mv
    ## Specify CDAT specific stuff ##
    # Set netcdf file criterion - turned on from default 0s
    cdm.setCompressionWarnings(0) ; # Suppress warnings
    cdm.setNetcdfShuffleFlag(0)
    cdm.setNetcdfDeflateFlag(1)
    cdm.setNetcdfDeflateLevelFlag(9)
    # Hi compression: 1.4Gb file ; # Single salt variable
    # No compression: 5.6Gb ; Standard (compression/shuffling): 1.5Gb ; Hi compression w/ shuffling: 1.5Gb
    cdm.setAutoBounds(1) ; # Ensure bounds on time and depth axes are generated
    ##
except:
    print('* cdat_info not available, skipping UV-CDAT import *')

## Define useful functions ##
 #%%
def cmipBranchTime(model,experiment,r1i1p1):
    """
    Documentation for cmipBranchTime():
    -------
    The cmipBranchTime() function returns simulation branch information for
    models that contributed to the CMIP5 historical experiment

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> from durolib import cmipBranchTime
        >>> branchTime = cmipBranchTime('ACCESS1-0','historical,'r1i1p1')

    Notes:
    -----
    - PJD 18 Jun 2018 - Implemented following existing historical database
    - Currently only contains CMIP5 historical simulation branch information
    """
    # Load dictionary into memory
    cmip5 = json.load(open(os.path.join(durolib_egg_path,'CMIP5BranchTimes.json'),'r'))
    # Validate user input
    cm5Models = ['model','ACCESS1-0','ACCESS1-3','BNU-ESM','CCSM4','CESM1-BGC',
                 'CESM1-CAM5','CESM1-CAM5-1-FV2','CESM1-FASTCHEM','CESM1-WACCM',
                 'CMCC-CESM','CMCC-CM','CMCC-CMS','CNRM-CM5','CSIRO-Mk3-6-0',
                 'CanESM2','EC-EARTH','FGOALS-g2','FGOALS-s2','FIO-ESM','GFDL-CM3',
                 'GFDL-ESM2M','GISS-E2-H-CC','GISS-E2-H','GISS-E2-R-CC','GISS-E2-R',
                 'HadGEM2-AO','HadGEM2-CC','HadGEM2-ES','IPSL-CM5A-LR','IPSL-CM5A-MR',
                 'IPSL-CM5B-LR','MIROC-ESM-CHEM','MIROC-ESM','MIROC4h','MIROC5',
                 'MPI-ESM-LR','MPI-ESM-MR','MPI-ESM-P','MRI-CGCM3','NorESM1-M',
                 'NorESM1-ME','bcc-csm1-1-m','bcc-csm1-1','inmcm4']
    cm5Experiments = ['historical','historicalNat']
    if model not in cm5Models:
        print(''.join(['Model: ',model,' not a valid CMIP5 contributor']))
        return None
    elif experiment not in cm5Experiments:
        print(''.join(['Experiment: ',experiment,' not currently indexed']))
        return None
    else:
        experiment = 'historical'
        branchInfo = cmip5.get('ACCESS1-0').get('historical')
        if r1i1p1 not in branchInfo.keys():
            print(''.join(['r1i1p1: ',r1i1p1,' not a valid ',model,' ',experiment,' simulation']))
            return None
        else:
            print(''.join(['Model: ',model,'; Exp: ',experiment,'; r1i1p1: ',r1i1p1,
                           ' branch times found']))
            branchInfoDict = branchInfo

    return branchInfoDict

#%%
def clearAll():
    """
    Documentation for clearall():
    -------
    The clearall() function purges all variables in global namespace

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> from durolib import clearAll
        >>> clearAll()

    Notes:
    -----
        Currently not working ...
    """
    for uniquevariable in [variable for variable in globals().copy() if variable[0] != "_" and variable != 'clearAll']:
        del globals()[uniquevariable]

#%%
def daysBetween(d1, d2):
    """
    Documentation for daysBetween():
    -------
    The daysBetween() function calculates days between two dates (strings)

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> from durolib import daysBetween
        >>> daysBetween('2015-11-17','2015-12-25')

    Notes:
    -----
        ...
    """
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

#%%
def environment():
    return False

#%%
def fillHoles(var):
    return var
    #http://tcl-nap.cvs.sourceforge.net/viewvc/tcl-nap/tcl-nap/library/nap_function_lib.tcl?revision=1.56&view=markup
    #http://tcl-nap.cvs.sourceforge.net/viewvc/tcl-nap/tcl-nap/library/stat.tcl?revision=1.29&view=markup
    #http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    #http://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
    #https://www.google.com/search?q=python+nearest+neighbor+fill
    """
     # fill_holes --
329 	#
330 	# Replace missing values by estimates based on means of neighbours
331 	#
332 	# Usage:
333 	# fill_holes(x, max_nloops)
334 	# where:
335 	# - x is array to be filled
336 	# - max_nloops is max. no. iterations (Default is to keep going until
337 	# there are no missing values)
338
339 	proc fill_holes {
340 	x
341 	{max_nloops -1}
342 	} {
343 	set max_nloops [[nap "max_nloops"]]
344 	set n [$x nels]
345 	set n_present 0; # ensure at least one loop
346 	for {set nloops 0} {$n_present < $n && $nloops != $max_nloops} {incr nloops} {
347 	nap "ip = count(x, 0)"; # Is present? (0 = missing, 1 = present)
348 	set n_present [[nap "sum_elements(ip)"]]
349 	if {$n_present == 0} {
350 	error "fill_holes: All elements are missing"
351 	} elseif {$n_present < $n} {
352 	nap "x = ip ? x : moving_average(x, 3, -1)"
353 	}
354 	}
355 	nap "x"
356 	}
    """

#%%
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
        print(''.join(['** fitPolynomial Error: >cubic fits not supported **',]))
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

#%%
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
    - PJD  6 Sep 2018 - Corrected time units 0-1-1 to 1-1-1
    -----
        ...
    """
    tind = range(shape(var)[0]) ; # Assume time axis is dimension 0
    t = cdm.createAxis(tind,id='time')
    t.units = 'years since 1-01-01 0:0:0.0'
    t.calendar = var.getTime().calendar
    cdu.times.setTimeBoundsYearly(t) ; # Explicitly set time bounds to yearly
    var.setAxis(0,t)
    return var

#%%
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
                print(''.join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
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
                print(''.join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
            if logFile is not None:
                writeToLog(logFile,"".join(['*SO mean:     {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
    elif varName in 'thetao':
        if var.max() > 50. and var.mean() > 265.:
            if report:
                print(''.join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
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
                print(''.join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))
            if logFile is not None:
                writeToLog(logFile,"".join(['*THETAO mean: {:+06.2f}'.format(var.mean()),'; min: {:+06.2f}'.format(var.min().astype('float64')),'; max: {:+06.2f}'.format(var.max().astype('float64'))]))

    return var,var_fixed

#%%
def getGitInfo(filePath):
    """
    Documentation for getGitInfo():
    -------
    The getGitInfo() function retrieves latest commit info specified by filePath

    Author: Paul J. Durack : pauldurack@llnl.gov

    Inputs:
    -----

    |  **filePath** - a fully qualified file which is monitored by git

    Returns:
    -------

    |  **gitTag[0]** - commit hash
    |  **gitTag[1]** - commit note
    |  **gitTag[2]** - commit tag_point (if numeric: x.x.x)
    |  **gitTag[3]** - commit date and time
    |  **gitTag[4]** - commit author

    Usage:
    ------
        >>> from durolib import getGitInfo
        >>> gitTag = getGitInfo(filePath)

    Notes:
    -----
    * PJD 26 Aug 2016 - Showing tags, see http://stackoverflow.com/questions/4211604/show-all-tags-in-git-log
    * PJD 26 Aug 2016 - Added tag/release info
    * PJD 31 Aug 2016 - Convert tag info to use describe function
    * PJD  1 Sep 2016 - Upgrade test logic
    * PJD 15 Nov 2016 - Broadened error case if not a valid git-tracked file
    * PJD 28 Nov 2016 - Tweaks to get git tags registering
    * PJD 17 Jul 2017 - Further work required to deal with tags which include '-' characters
    * PJD 13 Nov 2020 - Updated for Py3
    * PJD 20 Jun 2023 - Updated getGitInfo to test gitTagErr.strip() == b'str' vs 'str' for py3
    ...
    """
    # Test current work dir
    if os.path.isfile(filePath) or os.path.isdir(filePath):
        currentWorkingDir = os.path.split(filePath)[0]
        #os.chdir(currentWorkingDir) ; # Add convert to local dir
    else:
        print('filePath invalid, exiting')
        return ''
    # Get hash, author, dates and notes
    p = subprocess.Popen(['git','log','-n1','--',filePath],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         cwd=currentWorkingDir)
    stdout = p.stdout.read() ; # Use persistent variables for tests below
    stderr = p.stderr.read()
    if stdout == '' and stderr == '':
        print('filePath not a valid git-tracked file')
        return
    #elif 'fatal: ' in stderr: # Py2
    elif b'fatal: ' in stderr: # Py3
        print('filePath not a valid git-tracked file')
        return
    #gitLogFull = stdout # git full log # Py2
    gitLogFull = stdout.decode('utf-8') # git full log # Py3
    #print(gitLogFull)
    del(p)
    gitLog = []
    for count,gitStr in enumerate(gitLogFull.split('\n')):
        if gitStr == '':
            pass
        else:
            gitStr = gitStr.replace('   ',' ') ; # Trim excess whitespace in date
            gitStr = gitStr.replace('commit ','commit: ')
            if count < 3:
                gitStr = gitStr.strip()
                gitStr = gitStr[:1].lower() + gitStr[1:]
                gitLog.extend([gitStr])
                continue
            gitLog.extend([''.join(['note: ',gitStr.strip()])])

    # Get tag info
    #p = subprocess.Popen(['git','log','-n1','--no-walk','--tags',
    #                      '--pretty="%h %d %s"','--decorate=full',filePath],
    p = subprocess.Popen(['git','describe','--tags'],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          cwd=currentWorkingDir)
    #gitTag = p.stdout.read() # git tag log # Py2
    gitTag = p.stdout.read().decode('utf-8') # git tag log # Py3
    #print 'gitTag',gitTag
    gitTagErr = p.stderr.read() ; # Catch tag-less error
    #print 'gitTagErr',gitTagErr
    del(filePath,p)

    if gitTagErr.strip() == b'fatal: No names found, cannot describe anything.':
        gitLog.extend(['latest_tagPoint: None'])
    elif gitTag != '':
        # Example gitTag='CMOR-3.2.5\n' https://github.com/WCRP-CMIP/CMIP6_CVs/releases/tag/CMOR-3.2.5
        # Example gitTag='CMOR-3.2.5-42-gb07f789\n'
        for count,gitStr in enumerate(gitTag.split('\n')):
            #print 'count,gitStr',count,gitStr
            if gitStr == '':
                continue
            #hyphInds = []; ind = 0
            #while ind < len(gitStr):
            #    hyphInds.append(gitStr.rfind('-',ind))
            #    ind = gitStr.find('-',ind)
            tagBits = gitStr.split('-')
            #print tagBits
            tag = tagBits[0] ; # Doesn't work with 'CMOR-3.2.5\n'
            #print tag,len(tag)
            if gitTag.count('-') == 1: # Case tag point e.g. 'CMOR-3.2.5\n'
                tagPastCount = '0'
                tagHash = gitLog[0].replace('commit: ','')[0:7]
                tag = gitTag.strip('\n')
                gitLog.extend([''.join(['latest_tagPoint: ',tag,' (',tagPastCount,'; ',tagHash,')'])])
            elif gitTag.count('-') == 2: # Case beyond tag point
                tagPastCount = tagBits[1]
                tagHash = tagBits[2]
                tag = tagBits[0]
                gitLog.extend([''.join(['latest_tagPoint: ',tag,' (',tagPastCount,'; ',tagHash,')'])])
            elif gitTag.count('-') == 3: # Case beyond tag point
                tagPastCount = tagBits[2]
                tagHash = tagBits[3]
                tag = gitTag.split('\n')[0]
                gitLog.extend([''.join(['latest_tagPoint: ',tag,' (',tagPastCount,'; ',tagHash,')'])])
            else:
                gitLog.extend(['latest_tagPoint: ',tag])
    else:
        print('Tag retrieval error, exiting')
        print('gitTag:',gitTag,len(gitTag))
        print('gitTagErr:',gitTagErr,len(gitTagErr))
        return ''

    # Order list
    #print('gitLog:',gitLog)
    gitLog = [gitLog[0],gitLog[3],gitLog[4],gitLog[2],gitLog[1]]

    return gitLog

#%%
def globalAttWrite(file_handle,options):
    """
    Documentation for globalAttWrite():
    -------
    The globalAttWrite() function writes standard global_attributes to an
    open netcdf specified by file_handle

    Author: Paul J. Durack : pauldurack@llnl.gov

    Inputs:
    -----

    |  **file_handle** - a cdms2 open, writeable file handle

    Returns:
    -------
           Nothing.
    Usage:
    ------
        >>> from durolib import globalAttWrite
        >>> globalAttWrite(file_handle)

    Optional Arguments:
    -------------------
    |  option=optionalArguments
    |  Restrictions: option has to be a string
    |  Default : ...

    You can pass option='SOMETHING', ...

    Examples:
    ---------
        >>> from durolib import globalAttWrite
        >>> f = cdms2.open('data_file_name','w')
        >>> globalAttWrite(f)
        # Writes standard global attributes to the netcdf file specified by file_handle

    Notes:
    -----
    * PJD 19 Apr 2018 - Corrected UTC offset, now correct times are reported - https://github.com/stub42/pytz/issues/12
    """
    import cdat_info,pytz
    # Create timestamp, using UTC for history
    utcNow      = datetime.datetime.utcnow();
    utcNow      = utcNow.replace(tzinfo=pytz.utc)
    timeFormat  = utcNow.strftime("%d-%m-%Y %H:%M:%S %p")
    localTz     = pytz.timezone("America/Los_Angeles")
    localNow    = utcNow.astimezone(localTz)
    cdatVerInfo = cdat_info.version() ; # Deal with quirky formats
    if len(cdatVerInfo) > 2:
        cdatVerInfo = ".".join(["%s" % el for el in cdat_info.version()])
    else:
        cdatVerInfo = cdat_info.version()[-1].strip('v') ; # Trim off the v

    if 'options' in locals() and not options == None:
        if options.lower() == 'noid':
            file_handle.history = "".join(['File processed: ',timeFormat,' UTC; San Francisco, CA, USA'])
            file_handle.host    = "".join([gethostname(),'; CDAT version: ',cdatVerInfo,
                                                   '; Python version: ',sys.version.replace('\n','; ').replace(') ;',');')])
        else:
            print('** Invalid options passed, skipping global attribute write.. **')
    else:
        file_handle.data_contact    = "Paul J. Durack; pauldurack@llnl.gov; +1 925 422 5208"
        file_handle.history         = "".join(['File processed: ',timeFormat,' UTC; San Francisco, CA, USA'])

        file_handle.host            = "".join([gethostname(),'; CDAT version: ',cdatVerInfo,
                                               '; Python version: ',sys.version.replace('\n','; ').replace(') ;',');')])
        file_handle.institution     = "Program for Climate Model Diagnosis and Intercomparison (LLNL), Livermore, CA, U.S.A."

#%%
def globAndTrim(path):
    """
    Documentation for globAndTrim():
    -------
    The globAndTrim() function wraps trimModelList to take a single path argument

    Author: Paul J. Durack : pauldurack@llnl.gov

    Returns:
    -------
           List containing file paths
    Usage:
    ------
        >>> from durolib import globAndTrim
        >>> globAndTrim('/path/to/data')

    Examples:
    ---------
        ...

    Notes:
    -----
        ...
    """
    outList = glob.glob(os.path.join(path,'*.xml'))
    outList = trimModelList(outList) ; # only need to match model, no version info required
    outList.sort()
    return outList

#%%
def inpaint(array,method):
    #/work/durack1/csiro/Backup/110808/Z_dur041_linux/bin/inpaint_nans/inpaint_nans.m
    return False

#%%
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
    print('# Use quit() to exit :) Happy debugging!')
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner,local=namespace)
    except SystemExit:
        return

#%%
def makeCalendar(timeStart,timeEnd,calendarStep='months',monthStart=1,monthEnd=12,dayStep=1):
    """
    Documentation for makeCalendar():
    -----
    The makeCalendar() function creates a time calendar for given dates

    Author: Paul J. Durack : pauldurack@llnl.gov

    Inputs:
    -----

    |  **timeStart** - string start time (e.g. '2001' or '2001-1-1 0:0:0.0')
    |  **timeEnd** - string end time
    |  **calendarStep <optional>** - string either 'months' or 'days'
    |  **monthStart <optional>** - int
    |  **monthEnd <optional>** - int
    |  **dayStep <optional>** - int

    Returns:
    -----

    |  **time** - cdms2 transient axis

    Usage:
    -----
    >>> from durolib import makeCalendar
    >>> time = makeCalendar('2001','2014',calendarStep='months')

    Notes:
    -----
    * PJD 30 Apr 2015 - Fixed 'off by one' error with partial years
    * PJD  7 Jun 2016 - Corrected doc string for calendarStep argument
    * PJD  7 Jul 2016 - Fixed makeCalendar('1854','2016',monthStart='01',monthEnd='06',calendarStep='months') off by one error
    * PJD 25 Aug 2016 - Added 'years' calendarStep argument
    * TODO: Update to take full date identifier '2001-1-1 0:0:0.0', not just year
    * Issues with the daily calendar creation - likely require tweaks to cdutil.times.setAxisTimeBoundsDaily (range doesn't accept fractions, only ints)
    * Consider reviewing calendar assignment in /work/durack1/Shared/obs_data/AQUARIUS/read_AQ_SSS.py
    """
    # First check inputs
    if calendarStep not in ['days','months','years']:
        print('** makeCalendar error: calendarStep unknown, exiting..')
        return
    if calendarStep == 'years':
        monthStart = 1
        monthEnd = 12
    if not isinstance(timeStart,str) or not isinstance(timeEnd,str):
        print('** makeCalendar error: timeStart or timeEnd invalid, exiting..')
        return
    if not (int(monthStart) in range(1,13) and int(monthEnd) in range(1,13)):
        print('** makeCalendar error: monthStart or monthEnd invalid, exiting..')
        return
    try:
        timeStartTest   = cdt.comptime(int(timeStart))
        timeEndTest     = cdt.comptime(int(timeEnd))
    except SystemExit as err:
        print('** makeCalendar error: timeStart invalid - ',err)
        return

    # Create comptime objects
    if monthStart != 1:
        timeStart = cdt.comptime(int(timeStart),int(monthStart))
    else:
        timeStart = cdt.comptime(int(timeStart))
    test = re.compile('^[0-9]{4}$')
    if monthEnd == 2:
        if calendar.isleap(int(timeEnd)):
            timeEnd = cdt.comptime(int(timeEnd),2,29,23,59,59)
        else:
            timeEnd = cdt.comptime(int(timeEnd),2,28,23,59,59)
    elif monthEnd in [4,6,9,11]:
        timeEnd = cdt.comptime(int(timeEnd),int(monthEnd),30,23,59,59)
    else:
        timeEnd = cdt.comptime(int(timeEnd),int(monthEnd),31,23,59,59)
    # Set units for value conversion
    timeUnitsStr = ''.join([calendarStep,' since ',str(timeStart.year)])
    # Set times

    #print 'timeStart',timeStart
    #print 'timeEnd',timeEnd

    timeStart   = int(timeStart.torelative(timeUnitsStr).value)
    timeEnd     = int(timeEnd.torelative(timeUnitsStr).value)

    #print 'timeStart',timeStart
    #print 'timeEnd',timeEnd
    #print timeUnitsStr
    #print 'timeStart',cdt.r2c(timeStart,timeUnitsStr)
    #print 'timeEnd',cdt.r2c(timeEnd,timeUnitsStr)

    if 'dayStep' in locals() and calendarStep == 'days':
        times = np.float32(range(timeStart,timeEnd,dayStep)) ; # timeEnd/range requires +1 to reach end points (invalid correction 160707)
    else:
        times = np.float32(range(timeStart,timeEnd)) ; # timeEnd/range requires +1 to reach end points (invalid correction 160707)

    #print len(times)

    times                   = cdm.createAxis(times)

    #print times.shape

    times.designateTime()
    times.id                = 'time'
    times.units             = timeUnitsStr
    times.long_name         = 'time'
    times.standard_name     = 'time'
    times.calendar          = 'gregorian'
    times.axis              = 'T'

    #print times.asComponentTime()[0]
    #print times.asComponentTime()[-1]

    if calendarStep == 'days':
        #cdu.setTimeBoundsDaily(times,frequency=(1./dayStep))
        pass
    elif calendarStep == 'months':
        cdu.setTimeBoundsMonthly(times)
    elif calendarStep == 'years':
        cdu.setTimeBoundsYearly(times)
    #print times.asComponentTime()[0]
    #print times.asComponentTime()[-1]

    times.toRelativeTime(''.join(['days since ',str(times.asComponentTime()[0].year),'-1-1']))
    timeBounds  = times.getBounds()
    times[:]    = (timeBounds[:,0]+timeBounds[:,1])/2.
    #times[:]    = times[:]-1 ; # Correct off by one error (invalid correction 160707)

    return times

#%%
def matchAndTrimBlanks(varList,listFilesList,newVarId):
    """
    Documentation for matchAndTrimBlanks():
    -------
    The matchAndTrimBlanks() function takes a nested list of files, a
    corresponding list of variable names and a new variable name and returns a
    nested list containing matched files for differing variables

    Author: Paul J. Durack : pauldurack@llnl.gov

    Returns:
    -------
           Nested list containing input list and trimmed sublists
    Usage:
    ------
        >>> from durolib import matchAndTrimBlanks
        >>> soMatches = matchAndTrimBlanks(varList,listFilesList,newVarId)

    Examples:
    ---------
        >>> varList = ['so','tos','tas','wfo','areacello']
        >>> listFilesList = [so_fileList,tos_fileList,tas_fileList,wfo_fileList,fx_fileList]
        >>> newVarId = 'soMatches'
        >>> soMatches = matchAndTrimBlanks(varList,listFilesList,newVarId)

    Notes:
    -----
        ...
    """
    masterVar = varList[0]
    outSlots = len(varList)+1
    varMatchList = [[None] * outSlots for i in range(len(listFilesList[0][0][:]))]
    # For each model_noRealm in master List
    for x,modelNoRealm in enumerate(listFilesList[0][3][:]):
        varMatchList[x][0] = listFilesList[0][0][x]
        masterVarDot = ''.join(['.',masterVar,'.'])
        newVarDot = ''.join(['.',newVarId,'.'])
        # For each variable list - Pair masterVar with matches
        for y in range(0,len(varList)):
            # Test if fixed field - only match model
            if varList[y] in ['areacella','areacello','basin','deptho','orog','sftlf','sftof','volcello']:
                masterTest = modelNoRealm.split('.')[0]
                modOnly = True
            else:
                masterTest = modelNoRealm
                modOnly = False
            # Use try to deal with non-index
            try:
                modTest = []
                if modOnly:
                    for z,model in enumerate(listFilesList[y][3]):
                        modTest += [listFilesList[y][3][z].split('.')[0]]
                else:
                    modTest = listFilesList[y][3]
                index = modTest.index(masterTest)
                varMatchList[x][y] = listFilesList[y][0][index]
            except:
                print(format(x,'03d'),''.join(['No ',varList[y],' match for ',masterVar,': ',modelNoRealm]))
        # Create output fileName
        tmpStr = varMatchList[x][0].split('/')[-1]
        tmpStr = tmpStr.replace(masterVarDot,newVarDot).replace('.latestX.xml','.nc')
        varMatchList[x][outSlots-1] = tmpStr
        #varMatchList[x][outSlots-1] = replace(replace(varMatchList[x][0].split('/')[-1],masterVarDot,newVarDot),'.latestX.xml','.nc')
    return varMatchList

#%%
def mkDirNoOSErr(newdir,mode=777):
#def mkDirNoOSErr(newdir,mode=0777): Py3 doesn't accept leading zeros for numbers
    """
    Documentation for mkDirNoOSErr(newdir,mode=0777):
    -------
    The mkDirNoOSErr() function mimics os.makedirs however does not fail if the directory already
    exists

    Author: Paul J. Durack : pauldurack@llnl.gov

    Returns:
    -------
           Nothing.
    Usage:
    ------
        >>> from durolib import mkDirNoOSErr
        >>> mkDirNoOSErr('newPath',mode=0777)

    Notes:
    -----
    """
    try:
        os.makedirs(newdir,mode)
    except OSError as err:
        #Re-raise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise

#%%
def outerLocals(depth=0):
    return inspect.getouterframes(inspect.currentframe())[depth+1][0].f_locals

#%%
def readJsonCreateDict(buildList, urlPrefix=''):
    """
    Documentation for readJsonCreateDict(buildList, urlPrefix):
    -------
    The readJsonCreateDict() function reads web-based json files and writes
    their contents to a dictionary in memory

    Author: Paul J. Durack : pauldurack@llnl.gov

    The function takes a list argument with two entries. The first is the
    variable name for the assigned dictionary, and the second is the URL
    of the json file to be read and loaded into memory. Multiple entries
    can be included by generating additional embedded lists. An optional second
    argument urlPrefix can be used to simplify list expansion

    Usage:
    ------
        >>> from durolib import readJsonCreateDict
        >>> tmp = readJsonCreateDict([['Omon',
                                       'PCMDI/obs4MIPs-cmor-tables/master/Tables/obs4MIPs_Omon.json']],
                                       'https://raw.githubusercontent.com/')
        >>> Omon = tmp.get('Omon')

    Notes:
    -----
        PJD 24 Jul 2020  - Updated to take a second arg, the urlPrefix
        PJD 29 Jul 2020  - Updated to correct case when urlPrefix not supplied
        PJD 13 Nov 2020  - Updated for Py3
    """
    # Test for list input of length == 2
    if len(buildList[0]) != 2:
        print('Invalid inputs, exiting..')
        sys.exit()

    # Test for urlPrefix
    if urlPrefix != '':
        print('Using URLprefix:', urlPrefix)

    # Create urllib context to deal with lab/LLNL web certificates
    ctx                 = ssl.create_default_context()
    ctx.check_hostname  = False
    ctx.verify_mode     = ssl.CERT_NONE

    # Iterate through buildList and write results to jsonDict
    jsonDict = {}
    for count, table in enumerate(buildList):
        #print('Processing:', table[0])
        # Read web file
        if urlPrefix != '':
            url = ''.join([urlPrefix, table[1]])
            #print(url)
        else:
            url = table[1]
        #jsonOutput = urllib.urlopen(url, context=ctx) # Py2
        #jsonOutput = urlopen(table[1], context=ctx) # Py3
        #print('start')
        #print(url)
        #print(ctx)
        jsonOutput = urlopen(url, context=ctx) # Py3
        #print('end')
        tmp = jsonOutput.read()
        vars()[table[0]] = tmp
        jsonOutput.close()
        # Write local json
        tmpFile = open('tmp.json', 'w')
        #print(table[0])
        #tmpFile.write(eval(table[0])) # Py2
        tmpFile.write(eval(table[0]).decode('utf-8')) # Py3
        tmpFile.close()
        # Read local json
        vars()[table[0]] = json.load(open('tmp.json', 'r'))
        os.remove('tmp.json')
        jsonDict[table[0]] = eval(table[0]) # Write to dictionary

    return jsonDict

#%%
def santerTime(array,calendar=None):
        """
        Documentation for santerTime(array,calendar):
        -------
        The santerTime(array) function converts a known-time array to the
        standard time calendar - if non-gregorian the source calendar should
        be specified for accurate conversion

        Specified calendars can be one of the 5 calendars available within
        the cdtime module:
            GregorianCalendar
            MixedCalendar
            JulianCalendar
            NoLeapCalendar
            Calendar360
        For more information consult:
            http://cdat.llnl.gov/documentation/cdms/cdms_3.html#3.2

        Author: Paul J. Durack : pauldurack@llnl.gov

        Usage:
        ------
            >>> from durolib import santerTime
            >>> import cdtime
            >>> newVar = santerTime(var,calendar=cdtime.NoLeapCalendar)

        Notes:
        -----
        """
        # Test calendar
        if calendar:
            cdtCalendar  = calendar
        else:
            cdtCalendar  = cdt.GregorianCalendar
        # Set time_since - months 1800-1-1
        time                = array.getTime()
        time_new            = []
        for tt in time:
            reltime = cdt.reltime(tt,time.units)
            time_new.append(reltime.torel('months since 1800-1-1',cdtCalendar).value)
        time_axis           = cdm.createAxis(time_new)
        time_axis.id        = 'time'
        time_axis.units     = 'months since 1800-1-1'
        time_axis.axis      = 'T'
        time_axis.calendar  = 'gregorian'
        array.setAxis(0,time_axis)
        cdu.setTimeBoundsMonthly(array)
        return array

#%%
def scrubNaNAndMask(var,maskVar):
    """
    Documentation for scrubNaNAndMask(var,maskVar):
    -------
    The scrubNaNAndMask(var,maskVar) function determines NaN values within a
    numpy matrix and replaces these with 1e+20

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> from durolib import scrubNaNAndMask
        >>> noNanVar = scrubNaNAndMask(var,maskVar)

    Notes:
    -----
    """
    # Check for NaNs
    nanvals = isnan(var)
    var[nanvals] = 1e+20
    var = mv.masked_where(maskVar>=1e+20,var)
    var = mv.masked_where(maskVar.mask,var)
    return var

#%%
def smooth(array,method):
    #/apps/MATLAB/R2011b/toolbox/matlab/specgraph/smooth3.m
    #/apps/MATLAB/R2011b/toolbox/curvefit/curvefit/smooth.m
    return False

#%%
def sysCallTimeout(cmd,timeout):
    """
    Documentation for sysCallTimeout(cmd,timeout):
    -------
    The sysCallTimeout(cmd,timeout) function attempts to execute a system call
    (cmd) and times out in a specified time

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> from durolib import sysCallTimeout
        >>> sysCallTimeout(cmd,timeout)

    Notes:
    -----
    """
    start = time.time()
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    while time.time() - start < timeout:
        if p.poll() is not None:
            return
        time.sleep(0.1)
    p.kill()
    raise OSError('sysCallTimeout: System call timed out')

#%%
def trimModelList(modelFileList):
    """
    Documentation for trimModelList(modelFileList):
    -------
    The trimModelList(modelFileList) function takes a python list of model files
    and trims these for duplicates using file creation_date attribute along with
    temporal ordering info obtained from the file version identifier

    Author: Paul J. Durack : pauldurack@llnl.gov

    Usage:
    ------
        >>> modelFileList = glob.glob(os.path.join(filePath,'*.nc')) ; # provides full directory/file path
        >>> from durolib import trimModelList
        >>> modelFileListTrimmed = trimModelList(modelFileList)

    Notes:
    -----
    - PJD  1 Apr 2014 - Implement sanity checks for r1i1p1 matching for e.g.
    - PJD  1 Apr 2014 - Removed hard-coded ver- position
    - PJD  1 Apr 2014 - Added realisation test to ensure expected format
    - PJD 10 Oct 2018 - Updated to deal with new CMIPLib paths and filenames (CMIP6/CMIP5/cmip5 data)

    CMIP6:

    In [1]: from durolib import trimModelList
    In [2]: import glob
    In [3]: fileList = glob.glob('/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/*.xml')
    In [4]: fileListTrim = trimModelList(fileList)
    tmp: ['CNRM-CM6-1.piControl.r1i1p1f2.glb-l-gr', 'CNRM-CM6-1.piControl.r1i1p1f2.glb-p19-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-l-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-l-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-p19-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-p19-gr']
    Unique: ['CNRM-CM6-1.piControl.r1i1p1f2.glb-l-gr', 'CNRM-CM6-1.piControl.r1i1p1f2.glb-p19-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-l-gr', 'IPSL-CM6A-LR.piControl.r1i1p1f1.glb-p19-gr']
    Index: [[0], [1], [2, 3], [4, 5]]
    In [5]: fileList
    Out[5]:
    ['/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.CNRM-CERFACS.CNRM-CM6-1.r1i1p1f2.mon.hur.atmos.glb-l-gr.v20180814.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.CNRM-CERFACS.CNRM-CM6-1.r1i1p1f2.mon.hur.atmos.glb-p19-gr.v20180814.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-l-gr.v20180314.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-l-gr.v20180802.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-p19-gr.v20180314.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-p19-gr.v20180802.0000000.0.xml']
    In [6]: fileListTrim
    Out[6]:
    ['/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.CNRM-CERFACS.CNRM-CM6-1.r1i1p1f2.mon.hur.atmos.glb-l-gr.v20180814.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.CNRM-CERFACS.CNRM-CM6-1.r1i1p1f2.mon.hur.atmos.glb-p19-gr.v20180814.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-l-gr.v20180802.0000000.0.xml',
     '/work/cmip-dyn/CMIP6/CMIP/piControl/atmos/mon/hur/CMIP6.CMIP.piControl.IPSL.IPSL-CM6A-LR.r1i1p1f1.mon.hur.atmos.glb-p19-gr.v20180802.0000000.0.xml']

    CMIP5:

    In [1]: from durolib import trimModelList
    In [2]: import glob
    In [3]: fileList = glob.glob('/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.*.xml')
    In [4]: fileListTrim = trimModelList(fileList)
    In [5]: fileList
    Out[5]:
    ['/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.BCC.bcc-csm1-1-m.r1i1p1.mon.tas.atmos.glb-z1-gu.v20120709.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.BCC.bcc-csm1-1-m.r2i1p1.mon.tas.atmos.glb-z1-gu.v20120709.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.BCC.bcc-csm1-1-m.r3i1p1.mon.tas.atmos.glb-z1-gu.v20120709.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.BCC.bcc-csm1-1.r1i1p1.mon.tas.atmos.glb-z1-gu.v1.0000000.0.xml',
...
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.NSF-DOE-NCAR.CESM1-WACCM.r7i1p1.mon.tas.atmos.glb-z1-gu.v20140429.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.UNSW.CSIRO-Mk3L-1-2.r1i2p1.mon.tas.atmos.glb-z1-gu.v20170728.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.UNSW.CSIRO-Mk3L-1-2.r2i2p1.mon.tas.atmos.glb-z1-gu.v20170728.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.UNSW.CSIRO-Mk3L-1-2.r3i2p1.mon.tas.atmos.glb-z1-gu.v20170728.0000000.0.xml']
    In [6]: fileListTrim
    Out[6]:
    ['/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.CSIRO-BOM.ACCESS1-0.r1i1p1.mon.tas.atmos.glb-z1-gu.v20120727.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.CSIRO-BOM.ACCESS1-0.r2i1p1.mon.tas.atmos.glb-z1-gu.v20130726.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.CSIRO-BOM.ACCESS1-0.r3i1p1.mon.tas.atmos.glb-z1-gu.v20140402.0000000.0.xml',
...
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.BCC.bcc-csm1-1.r3i1p1.mon.tas.atmos.glb-z1-gu.v1.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.FIO.fio-esm.r1i1p1.mon.tas.atmos.glb-z1-gu.v20121010.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.FIO.fio-esm.r2i1p1.mon.tas.atmos.glb-z1-gu.v20120522.0000000.0.xml',
     '/work/cmip-dyn/CMIP5/CMIP/historical/atmos/mon/tas/CMIP5.CMIP.historical.INM.inmcm4.r1i1p1.mon.tas.atmos.glb-z1-gu.1.0000000.0.xml']


    cmip5:

In [1]: from durolib import trimModelList
In [2]: import glob
In [3]: fileList = glob.glob('/work/cmip5-test/historical/atm/mo/tas/*.xml')
In [4]: len(fileList)
Out[4]: 467
In [5]: fileListTrim = trimModelList(fileList)
In [6]: len(fileListTrim)
Out[6]: 220
In [7]: fileList
Out[7]:
['/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-1.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-v1.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-v20120329.latestX.xml',
...
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.fio-esm.historical.r2i1p1.mo.atm.Amon.tas.ver-v20120522.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.inmcm4.historical.r1i1p1.mo.atm.Amon.tas.ver-1.latestX.xml']
In [8]: fileListTrim
Out[8]:
['/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-1.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r2i1p1.mo.atm.Amon.tas.ver-v20130726.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.ACCESS1-0.historical.r3i1p1.mo.atm.Amon.tas.ver-v20140402.latestX.xml',
...
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.bcc-csm1-1.historical.r3i1p1.mo.atm.Amon.tas.ver-v1.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.fio-esm.historical.r1i1p1.mo.atm.Amon.tas.ver-v20121010.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.fio-esm.historical.r2i1p1.mo.atm.Amon.tas.ver-v20120522.latestX.xml',
 '/work/cmip5-test/historical/atm/mo/tas/cmip5.inmcm4.historical.r1i1p1.mo.atm.Amon.tas.ver-1.latestX.xml']

    """
    # Check for list variable
    if type(modelFileList) is not list:
        print('** Function argument not type list, exiting.. **')
        return ''

    # Sort list and declare output
    modelFileList.sort()
    modelFileListTmp = []
    modelFileIndex = []

    # Create test defaults
    cmip5,cmip6 = [False for _ in range(2)]

    # Create subset modelFileList
    for file1 in modelFileList:
        file1   = file1.split('/')[-1]
        if 'cmip5' == file1.split('.')[0]:
            cmip5 = True
            #cmip5-test/historical/atm/mo/tas/
            #cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-1.latestX.xml
            #cmip5.ACCESS1-0.historical.r1i1p1.mo.atm.Amon.tas.ver-v20120329.latestX.xml
            mod     = file1.split('.')[1]
            exp     = file1.split('.')[2]
            rea     = file1.split('.')[3]
            gridLab = 'x'
            # Test rea for r1i1p111 format match
            reaTest = re.compile('^r\d{1,2}i\d{1,2}p\d{1,3}')
        elif file1.split('.')[0] in ['CMIP5','CMIP6']:
            cmip6 = True
            #cmip-dyn/CMIP6/CMIP/historical/atmos/mon/tas/
            #CMIP6.CMIP.historical.NOAA-GFDL.GFDL-CM4.r1i1p1f1.mon.tas.atmos.glb-z1-gr1.v20180301.0000000.0.xml
            mod     = file1.split('.')[4]
            #print('mod:',mod)
            exp     = file1.split('.')[2]
            #print('exp:',exp)
            rea     = file1.split('.')[5]
            #print('rea:',rea)
            gridLab = file1.split('.')[9]
            #print('gridLab:',gridLab)
            # Test rea for r1i1pf1 format match
            if file1.split('.')[0] == 'CMIP5':
                reaTest = re.compile('^r\d{1,2}i\d{1,2}p\d{1,3}')
            elif file1.split('.')[0] == 'CMIP6':
                reaTest = re.compile('^r\d{1,2}i\d{1,2}p\d{1,3}f\d{1,3}')

        # Evaluate components
        if not reaTest.match(rea):
            #pdb.set_trace()
            print('** Filename format invalid - rea: ',rea,', exiting.. **')
            return ''
        modelFileListTmp.append('.'.join([mod,exp,rea,gridLab]))

    # Create unique list and index
    modelFileListTmpUnique = list(set(modelFileListTmp)) ; modelFileListTmpUnique.sort()
    findMatches = lambda searchList,elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
    modelFileListTmpIndex = findMatches(modelFileListTmp,modelFileListTmpUnique)
    #print 'tmp:',modelFileListTmp
    #print 'Unique:',modelFileListTmpUnique
    #print 'Index:',modelFileListTmpIndex

    # Loop through unique list
    for count,modelNum in enumerate(modelFileListTmpIndex):
        if len(modelFileListTmpIndex[count]) == 1: # Case single version
            modelFileIndex.append(int(str(modelNum).strip('[]')))
        else: # Case multiple versions
            # Get version and creation_date info from file
            modelFileListVersion = [] ; modelFileListCreationDate = [] ; modelFileListIndex = []
            for index in modelFileListTmpIndex[count]:
                f_h = cdm.open(modelFileList[index])
                if cmip5:
                    file1 = modelFileList[index].split('/')[-1]
                    #print file1
                    #print 'new:',str([count for count,x in enumerate(file1.split('.')) if 'ver-' in x]).strip('[]')
                    verInd = int(str([count for count,x in enumerate(file1.split('.')) if 'ver-' in x]).strip('[]'))
                    ver1 = file1.split('.')[verInd].replace('ver-','')
                    CD = f_h.creation_date
                    modelFileListVersion.append(ver1)
                    modelFileListCreationDate.append(CD)
                    modelFileListIndex.append(index)
                elif cmip6:
                    ver1 = file1.split('.')[10]
                    CD = f_h.creation_date
                    modelFileListVersion.append(ver1)
                    modelFileListCreationDate.append(CD)
                    modelFileListIndex.append(index)
                else:
                    print('** mip_era unidentified, exiting.. **')
                f_h.close()
            #print modelFileListVersion
            #print modelFileListCreationDate

            # Use creation_date to determine latest file
            listLen = len(modelFileListCreationDate)
            modelFileListCreationDate = map(string.replace,modelFileListCreationDate,['T',]*listLen, [' ',]*listLen)
            modelFileListCreationDate = map(cdt.s2c,modelFileListCreationDate)
            modelFileListCreationDate = map(cdt.c2r,modelFileListCreationDate,['days since 1-1-1',]*listLen)
            modelFileListCreationDate = [x.value for x in modelFileListCreationDate]
            maxes = [i for i,x in enumerate(modelFileListCreationDate) if x == max(modelFileListCreationDate)]
            ver = [modelFileListVersion[i] for i in maxes]
            ind = [modelFileListIndex[i] for i in maxes]
            #print modelFileListCreationDate
            #print maxes,ver,ind

            # If creation_dates match check version info to determine latest file
            indTest = '-' ; #verTest = '-'
            if len(maxes) > 1:
                pubTest = 0 ; dateTest = 0;
                for count,ver1 in reversed(list(enumerate(ver))):
                    # Take datestamp versioned data
                    if 'v' in ver1 and ver1 > dateTest:
                        #verTest = ver[count]
                        indTest = ind[count]
                        dateTest = ver1
                    # Use published data preferentially: 1,2,3,4, ...
                    if ver1.isdigit() and ver1 > pubTest:
                        indTest = ind[count]
                        pubTest = ver1
                modelFileIndex.append(int(str(indTest).strip('[]')))
            else:
                modelFileIndex.append(int(str(ind).strip('[]')))
            #print pubTest,dateTest

    # Trim original list with new index
    modelFileListTrimmed = [modelFileList[i] for i in modelFileIndex]

    #return modelFileListTrimmed,modelFileIndex,modelFileListTmp,modelFileListTmpUnique,modelFileListTmpIndex ; # Debugging
    return modelFileListTrimmed

#%%
def truncateVerInfo(fileList,varId,modelSuite):
    """
    Documentation for truncateVerInfo():
    -------
    The truncateVerInfo() function takes a list of files and returns a nested
    list containing trimmed sublists

    Author: Paul J. Durack : pauldurack@llnl.gov

    Returns:
    -------
           Nested list containing input list and trimmed sublists
    Usage:
    ------
        >>> from durolib import truncateVerInfo
        >>> truncateVerInfo(fileList,'so','cmip5')

    Examples:
    ---------
        ...

    Notes:
    -----
        ...
    """
    fileList_noVar,fileList_noVer,fileList_noRealm = [[] for _ in range(3)]
    varId = ''.join(['.',varId])
    for infile in fileList:
        #tmp = replace(replace(replace(infile.split('/')[-1],varId,''),''.join([modelSuite,'.']),''),'.latestX.xml','')
        tmp = infile.split('/')[-1].replace(varId,'').replace(''.join([modelSuite,'.']),'').replace('.latestX.xml','')
        fileList_noVar += [tmp];
        tmp = '.'.join(tmp.split('.')[0:-1]) ; # truncate version info
        fileList_noVer += [tmp]
        tmp = '.'.join(tmp.split('.')[0:-3]) ; # truncate realm and temporal info
        fileList_noRealm += [tmp]
    return [fileList,fileList_noVar,fileList_noVer,fileList_noRealm]

#%%
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

#%%
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
        TODO: deal with input data precision
    """
    #varType             = var.dtype
    varMin              = var.min()
    varMax              = var.max()
    var.scale_factor    = np.float32((varMax-varMin)/(2**16)-1)
    var.add_offset      = np.float32(varMin+var.scale_factor*(2**15))
    if 'tmp.nc' in fileObject:
        fileObject = cdm.open(fileObject,'w')
    fileObject.write((var-var.add_offset)/var.scale_factor,dtype=np.int16)
    return
