#!/usr/bin/env python

import sys
sys.path.append('/data/mstid/statistics/webserver')

import os
import inspect
import datetime
from hdf5_api import loadMusicArrayFromHDF5
import shutil
from operator import itemgetter
import glob

import numpy as np
from scipy import stats as stats
from scipy.io import readsav
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from mpl_toolkits.basemap import Basemap

import sklearn.mixture

import pymongo
from bson.objectid import ObjectId

import utils

import music_support as msc
import stats_support as ssup


curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

font = {'weight':'bold', 'size':16}
matplotlib.rc('font', **font)

# Get events from mongo database. ##############################################
mongo         = pymongo.MongoClient()
db            = mongo.mstid


## Settings used for originial BKS Automated Run ################################
#radar = 'bks'
##mstid_list = 'manual_list'
##mstid_list = 'MUSIC Processed Round 3'
## I believe automated_run_1_bks_0817mlt is the same as automated_run_1, except that it only has bks from 08-17mlt
#mstid_list = 'automated_run_1_bks_0817mlt'
#
##sTime = datetime.datetime(2010,6,1)
#sTime = datetime.datetime(2010,7,29)
#eTime = datetime.datetime(2011,6,1)
#
#min_mlt = 8
#max_mlt = 17
#
#beamLimits_0            = None
#beamLimits_1            = None
#gateLimits_0            = 10
#gateLimits_1            = 35
#################################################################################

## Settings used for FHE Automated Run ##########################################
#radar = 'fhe'
#mstid_list = 'fhe_automated_run_1'
#
#sTime = datetime.datetime(2011,11,23)
#eTime = datetime.datetime(2012,1,1)
#
#min_mlt = 7
#max_mlt = 17
#
#beamLimits_0            = None
#beamLimits_1            = None
#gateLimits_0            = 15
#gateLimits_1            = 45
#################################################################################

## Settings used for FHE Automated Run ##########################################
#mstid_list = 'bks_2012_automated_run'
##sTime = datetime.datetime(2012,1,20,16)
#
#beamLimits_0            = None
#beamLimits_1            = None
#gateLimits_0            = 12
#gateLimits_1            = 45
#################################################################################

# Runs for Paper ###############################################################
rti_only    = False
no_replot   = True
#mstid_list = 'paper2_bks'
#gateLimits_0,gateLimits_1 =  (10,50)

mstid_list = 'paper2_wal'
gateLimits_0,gateLimits_1 =  (10,50)

#mstid_list = 'paper2_fhe'
#gateLimits_0,gateLimits_1 =  (10,50)

#mstid_list = 'paper2_fhw'
#gateLimits_0,gateLimits_1 =  (10,50)

#mstid_list = 'paper2_cve'
#gateLimits_0,gateLimits_1 =  (10,50)

#mstid_list = 'paper2_cvw'
#gateLimits_0,gateLimits_1 =  (10,50)
################################################################################

mstidDayDict,quietDayDict,noneDayDict = ssup.loadDayLists(mstid_list=mstid_list)

#allEvents = mstidDayDict+quietDayDict+noneDayDict 
allEvents = mstidDayDict

for event in allEvents:
    if 'signals' in event:
        _id = event['_id']
        status = db[mstid_list].update({'_id':_id},{'$unset': {'signals': 1}})

# Error file handling. #########################################################
error_file_path = 'static/music/automated_error.txt'
try:
    os.remove(error_file_path)
except:
    pass

for event in allEvents:
    radar                   = str(event['radar'])
    sDatetime               = event['sDatetime']
    fDatetime               = event['fDatetime']

    try:
        if sDatetime < datetime.datetime(2013,3,1):
            print('Skipping: ', radar, sDatetime)
            continue
    except:
        pass

    try:
        if sDatetime < sTime:
            print('Skipping: ', radar, sDatetime)
            continue
    except:
        pass

    musicPath   = msc.get_output_path(radar, sDatetime, fDatetime)
    rtiPath     = os.path.join(musicPath,'000_originalFit_RTI.png')
    if rti_only and no_replot:
        if os.path.exists(rtiPath):
            print('Already plotted: ', radar, sDatetime)
            continue

    print('Processing: ', radar, sDatetime)

    try:
        interpolationResolution = 120.
        filterNumtaps           = 101.
        firFilterLimits_0       = 0.0003
        firFilterLimits_1       = 0.0012
        window_data             = False
        kx_max                  = 0.05
        ky_max                  = 0.05
        autodetect_threshold_str = 0.35
        neighborhood_0          = 10
        neighborhood_1          = 10

        try:
            bl0 = int(beamLimits_0)
        except:
            bl0 = None
        try:
            bl1 = int(beamLimits_1)
        except:
            bl1 = None
        beamLimits = (bl0, bl1)

        try:
            gl0 = int(gateLimits_0)
        except:
            gl0 = None
        try:
            gl1 = int(gateLimits_1)
        except:
            gl1 = None
        gateLimits = (gl0,gl1)

        try:
            interpRes = int(interpolationResolution)
        except:
            interpRes = None

        try:
            numtaps = int(filterNumtaps)
        except:
            numtaps = None

        try:
            cutoff_low  = float(firFilterLimits_0)
        except:
            cutoff_low  = None

        try:
            cutoff_high  = float(firFilterLimits_1)
        except:
            cutoff_high  = None

        try:
            kx_max  = float(kx_max)
        except:
            kx_max  = 0.05

        try:
            ky_max  = float(ky_max)
        except:
            ky_max  = 0.05

        try:
            autodetect_threshold  = float(autodetect_threshold_str)
        except:
            autodetect_threshold  = 0.35

        try:
            nn0 = int(neighborhood_0)
        except:
            nn0 = None
        try:
            nn1 = int(neighborhood_1)
        except:
            nn1 = None
        neighborhood = (nn0,nn1)

        ################################################################################ 
        try:
            shutil.rmtree(musicPath)
        except:
            pass

        dataObj = msc.createMusicObj(radar.lower(), sDatetime, fDatetime
            ,beamLimits                 = beamLimits
            ,gateLimits                 = gateLimits
            ,interpolationResolution    = interpRes
            ,filterNumtaps              = numtaps 
            )

        hdf5Path  = msc.get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)


        # Create a run file. ###########################################################
        runParams = {}
        runParams['radar']              = radar.lower()
        runParams['sDatetime']          = sDatetime
        runParams['fDatetime']          = fDatetime
        runParams['beamLimits']         = beamLimits
        runParams['gateLimits']         = gateLimits
        runParams['interpRes']          = interpRes
        runParams['filter_numtaps']     = numtaps
        runParams['filter_cutoff_low']  = cutoff_low
        runParams['filter_cutoff_high'] = cutoff_high
        runParams['path']               = musicPath
        runParams['musicObj_path']      = hdf5Path
        runParams['window_data']        = window_data
        runParams['kx_max']             = kx_max
        runParams['ky_max']             = ky_max
        runParams['autodetect_threshold'] = autodetect_threshold
        runParams['neighborhood']        = neighborhood

        runfile_obj = msc.Runfile(radar.lower(), sDatetime, fDatetime, runParams)
        runfile_path    = runfile_obj.runParams['runfile_path']

        # Generate general RTI plot for original data. #################################
        rti_xlim    = msc.get_default_rti_times(runParams,dataObj)
        rti_ylim    = msc.get_default_gate_range(runParams,dataObj)
        rti_beams   = msc.get_default_beams(runParams,dataObj)
        dataObj.DS000_originalFit.metadata['timeLimits'] = [runParams['sDatetime'],runParams['fDatetime']]
        msc.plot_music_rti(dataObj,fileName=rtiPath,dataSet="originalFit",beam=rti_beams,xlim=rti_xlim,ylim=rti_ylim)
        dataObj.DS000_originalFit.metadata.pop('timeLimits',None)
        if not rti_only: 
            msc.run_music(runfile_path)
#           msc.music_plot_all(runfile_path)

            # Add all signals to database. #################################################
            dataObj = loadMusicArrayFromHDF5(hdf5Path)
            dataSets    = dataObj.get_data_sets()
            currentData = getattr(dataObj,dataSets[-1])

            #Pull up database record to see if there are already items stored.
            _id = event['_id']

            if 'signals' in event:
                event.pop('signals',None)

            sigList = []
            serialNr    = 0

            if hasattr(currentData,'sigDetect'):
                sigs    = currentData.sigDetect
                sigs.reorder()
                for sig in sigs.info:
                    sigInfo = {}
                    sigInfo['order']    = int(sig['order'])
                    sigInfo['kx']       = float(sig['kx'])
                    sigInfo['ky']       = float(sig['ky'])
                    sigInfo['k']        = float(sig['k'])
                    sigInfo['lambda']   = float(sig['lambda'])
                    sigInfo['azm']      = float(sig['azm'])
                    sigInfo['freq']     = float(sig['freq'])
                    sigInfo['period']   = float(sig['period'])
                    sigInfo['vel']      = float(sig['vel'])
                    sigInfo['max']      = float(sig['max'])
                    sigInfo['area']     = float(sig['area'])
                    sigInfo['serialNr'] = serialNr
                    sigList.append(sigInfo)
                    serialNr = serialNr + 1

            status = db[mstid_list].update({'_id':_id},{'$set': {'signals':sigList}})
    except:
        with open(error_file_path,'a') as error_file:
            error_file.write(' '.join([event['radar'],str(event['sDatetime']),str(event['fDatetime'])])+'\n')
