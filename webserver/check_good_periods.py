#!/usr/bin/env python
#import sys
#sys.path.append('/data/mstid/statistics')

import os
import datetime
import pickle
import shutil
from operator import itemgetter
import glob

from pyDARNmusic import checkDataQuality
# import numpy as np
from scipy import stats as stats
# from scipy.io import readsav

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt


import music_support as msc
from auto_range import *
#from ocv_edge_detect import *


import pymongo
mongo   = pymongo.MongoClient()
db      = mongo.mstid

################################################################################
def events_from_mongo(mstid_list,sDate,eDate,category=None):
    """Allow connection to mongo database."""

    crsr    = db[mstid_list].find({'date': {'$gte': sDate, '$lt': eDate}})
    event_list  = []
    for item in crsr:
        if category:
            if category == 'none':
                if 'category_manu' in item:
                    if item['category_manu'].lower() != 'none': continue
            if category == 'mstid':
                if 'category_manu' in item:
                    if item['category_manu'].lower() != 'mstid': continue
                else: continue
            if category == 'quiet':
                if 'category_manu' in item:
                    if item['category_manu'].lower() != 'quiet': continue
                else: continue
        else:
            category = 'all'

        tmp = {}
        tmp['radar']        = str(item['radar'])
        tmp['sDatetime']    = item['sDatetime']
        tmp['fDatetime']    = item['fDatetime']
        tmp['category']     = category
        tmp['_id']          = item['_id']
        if 'category_manu' in item:
            tmp['db_category'] = item['category_manu']
        else:
            tmp['db_category'] = 'None'
        event_list.append(tmp)

    event_list  = sorted(event_list,key=lambda k: k['sDatetime'])
    return event_list

def run_music(event_list,
    mstid_list                   = 'Undefined',
    new_music_obj               = True,
    only_plot_rti               = False,
    process_level               = 'all',
    make_plots                  = True,
    clear_output_dir            = False,
    output_dir                  = 'output',
    error_file_path             = 'static/music/automated_error.txt',
    default_beam_limits         = (None, None),
    default_gate_limits         = (0,80),
    auto_range_on               = True,
    interpolationResolution     = 120.,
    filterNumtaps               = 101.,
    firFilterLimits_0           = 0.0003,
    firFilterLimits_1           = 0.0012,
    window_data                 = False,
    kx_max                      = 0.05,
    ky_max                      = 0.05,
    autodetect_threshold_str    = 0.35,
    neighborhood_0              = 10,
    neighborhood_1              = 10):
    """
    new_music_obj       = True  #Generate a completely fresh dataObj file from fitex files.
    only_plot_rti       = False
    clear_output_dir    = False #Remove an old output directory if it exists.
    """

    nr_events   = len(event_list)
    event_nr    = 0

    event_counter   = {}
    for key in ['mstid','quiet','None']:
        event_counter[key]  = {}
        event_counter[key]['total'] = 0
        event_counter[key]['good']  = 0

    event_counter['error_count'] = 0
    error_list  = []

    # Run through actual events. ###################################################
    for event in event_list:
        try:
            event_nr                += 1
            radar                   = str(event['radar'])
            sDatetime               = event['sDatetime']
            fDatetime               = event['fDatetime']

            ################################################################################ 
            musicPath   = msc.get_output_path(radar, sDatetime, fDatetime)
            picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)

            if 'category' in event:
                print_cat = '(%s)' % event['category']
            else:
                print_cat = ''
            now = datetime.datetime.now()

            print(now,print_cat,'(%d of %d)' % (event_nr, nr_events), 'Processing: ', radar, sDatetime)
            if os.path.exists(picklePath):
                dataObj     = pickle.load(open(picklePath,'rb'))
            else:

                ################################################################################ 
                if 'beam_limits' in event:
                    beamLimits_0            = event['beam_limits'][0]
                    beamLimits_1            = event['beam_limits'][1]
                else:
                    beamLimits_0            = default_beam_limits[0]
                    beamLimits_1            = default_beam_limits[1]

                if 'gate_limits' in event:
                    gateLimits_0            = event['gate_limits'][0]
                    gateLimits_1            = event['gate_limits'][1]
                else:
                    gateLimits_0            = default_gate_limits[0]
                    gateLimits_1            = default_gate_limits[1]


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

                dataObj = msc.createMusicObj(radar.lower(), sDatetime, fDatetime
                    ,beamLimits                 = beamLimits
                    ,gateLimits                 = gateLimits
                    ,interpolationResolution    = interpRes
                    ,filterNumtaps              = numtaps 
                    )

            dataObj = checkDataQuality(dataObj,dataSet='DS000_originalFit',max_off_time=10,sTime=sDatetime,eTime=fDatetime)
            good    = dataObj.DS000_originalFit.metadata['good_period']
            if good: 
                event_counter[event['db_category']]['good'] += 1
            event_counter[event['db_category']]['total'] += 1

            db[mstid_list].update({'_id':event['_id']},{'$set':{'good_period':good}})

        except:
            now = str(datetime.datetime.now())+':'
            err =' '.join([now,event['radar'],str(event['sDatetime']),str(event['fDatetime'])])+'\n'
            print('CHECK_GOOD_PERIOD ERROR: '+err)
            db[mstid_list].update({'_id':event['_id']},{'$set':{'good_period':False}})

            error_list.append(err)
            event_counter['error_count'] += 1

    report_file = os.path.join('static','music',mstid_list+'.txt')
    with open(report_file,'w') as fl:
        fl.write(str(datetime.datetime.now()))
        fl.write('\n')
        fl.write(report_file)
        fl.write('\n')

        fl.write(str(event_counter))
        fl.write('\n')
        fl.write('\n')
        fl.write('\n'.join(error_list))
    return event_counter


if __name__ == '__main__':
    # Database Connection ########################################################## 
    mstid_list  = 'paper2_wal_2012'
    #mstid_list  = 'paper2_bks_2012'
#    mstid_list  = 'paper2_cvw_2012'
    sDate       = datetime.datetime(2012,11,1)
    eDate       = datetime.datetime(2013,1,1)
    event_list  = events_from_mongo(mstid_list,sDate,eDate,category='all')

#    event_list   = []
##    event_list.append({'radar':'wal', 'sDatetime':datetime.datetime(2012,11,7,20),'fDatetime':datetime.datetime(2012,11,7,22)})
#    event_list.append({'radar':'wal', 'sDatetime':datetime.datetime(2012,12,11,20),'fDatetime':datetime.datetime(2012,12,11,22)})
    run_music(event_list,mstid_list=mstid_list)
