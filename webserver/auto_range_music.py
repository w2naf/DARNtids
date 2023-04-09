#!/usr/bin/env python
#import sys
#sys.path.append('/data/mstid/statistics')

import os
import datetime
import pickle
import shutil
from operator import itemgetter
import glob

import numpy as np
from scipy import stats as stats
from scipy.io import readsav

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import traceback

from davitpy import utils
from davitpy import pydarn

import music_support as msc
from auto_range import *
#from ocv_edge_detect import *
from musicRTI3 import musicRTI3

import multiprocessing

def check_already_computed(q,event,quick_check=True):
    radar       = event['radar']
    sDatetime   = event['sDatetime']
    fDatetime   = event['fDatetime']

    picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)
    if os.path.exists(picklePath):
        if quick_check:
            event = None
        else:
            try:
                with open(picklePath,'rb') as fl:
                    test_obj    = pickle.load(fl)
                if not hasattr(test_obj,'active'):
                    pass
                elif hasattr(test_obj.active,'sigDetect'):
                    now = datetime.datetime.now()
                    event = None
            except:
                print 'Bad file: {0}'.format(picklePath)
#    return event
    q.put(event)

################################################################################
def events_from_mongo(mstid_list,sDate=None,eDate=None,category=None,months=None,recompute=False):
    """Allow connection to mongo database."""
    import pymongo
    mongo   = pymongo.MongoClient()
    db      = mongo.mstid

    query_dict  = {'date':{}}
    if sDate is not None:
        query_dict['date']['$gte'] = sDate

    if eDate is not None:
        query_dict['date']['$lt']  = eDate

    if sDate is not None and eDate is not None:
        crsr    = db[mstid_list].find(query_dict)
    else:
        crsr    = db[mstid_list].find()

    event_list  = []
    for item in crsr:
        if category:
            this_cat = item.get('category_manu')

            if category.lower() == 'unclassified':
                if this_cat is not None: continue
            if category.lower() == 'none':
                if this_cat != 'none': continue
            if category.lower() == 'mstid':
                if this_cat != 'mstid': continue
            if category.lower() == 'quiet':
                if this_cat != 'quiet': continue
        else:
            category = 'all'

        if months is not None:
            if item['sDatetime'].month not in months:
                continue

        tmp = {}
        tmp['radar']        = str(item['radar'])
        tmp['sDatetime']    = item['sDatetime']
        tmp['fDatetime']    = item['fDatetime']
        tmp['category']     = category
        event_list.append(tmp)

    event_list  = sorted(event_list,key=lambda k: k['sDatetime'])
    
    if not recompute:
        event_list_1 = []
        for event in event_list:
            print event
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=check_already_computed, args=(queue, event))
            p.start()
            p.join() # this blocks until the process terminates
            result = queue.get()
            if result is not None: event_list_1.append(result)
        event_list = event_list_1

    return event_list

def check_list_for_bad_start_gate(event_list):
    #This was done because auto_range originally allowed for selection of range gates
    #too close to the radar.
    recompute_list = []
    event_nr    = 0
    for event in event_list:
            event_nr                += 1
            radar                   = str(event['radar'])
            sDatetime               = event['sDatetime']
            fDatetime               = event['fDatetime']

            picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)
            with open(picklePath,'rb') as fl:
                dataObj = pickle.load(fl)

            bad_range   = np.max(np.where(dataObj.DS000_originalFit.fov.slantRCenter < 0)[1])
            if np.min(dataObj.DS000_originalFit.metadata['gateLimits']) <= bad_range:
                recompute_list.append(event)

    return recompute_list

def run_music(event_list,
    fitfilter                   = True,
    new_music_obj               = True,
    only_plot_rti               = False,
#    process_level               = 'all',
    process_level               = 'fft',
    make_plots                  = True,
    clear_output_dir            = False,
    output_dir                  = 'output',
    error_file_path             = 'static/music/automated_error.txt',
    default_beam_limits         = (None, None),
    default_gate_limits         = (0,80),
    auto_range_on               = True,
    interpolationResolution     = 60.,
#    filterNumtaps               = None,
#    firFilterLimits_0           = None,
#    firFilterLimits_1           = None,
    filterNumtaps               = 101.,
    firFilterLimits_0           = 0.0003,
    firFilterLimits_1           = 0.0012,
#    window_data                 = True,
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

    # Define show-all script.  Used multiple times throughout this code. ###########
    show_all_txt = ''' 
    <?php
    foreach (glob("*.png") as $filename) {
    echo "<img src='$filename'>";
    }
    ?>
    '''

    # Set-up output directories. ###################################################
    output_dirs             = {}
    output_dirs['range']    = os.path.join(output_dir,'range_distribution')
#    output_dirs['edge']     = os.path.join(output_dir,'edge_detection')

    for key in output_dirs:
        if clear_output_dir:
            try:
                shutil.rmtree(output_dirs[key])
            except:
                pass
        if not os.path.exists(output_dirs[key]): os.makedirs(output_dirs[key])
        with open(os.path.join(output_dirs[key],'0000-show_all.php'),'w') as file_obj:
            file_obj.write(show_all_txt)

    event_list  = np.array(event_list)
    if event_list.shape == (): event_list.shape = (1,)

    nr_events   = len(event_list)
    event_nr    = 0
    # Run through actual events. ###################################################
    for event in event_list:
        plt.close('all')
        try:
            event_nr                += 1
            radar                   = str(event['radar'])
            sDatetime               = event['sDatetime']
            fDatetime               = event['fDatetime']

            ################################################################################ 
            musicPath   = msc.get_output_path(radar, sDatetime, fDatetime)
            picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)

            if event.has_key('category'):
                print_cat = '(%s)' % event['category']
            else:
                print_cat = ''
            now = datetime.datetime.now()

            print now,print_cat,'(%d of %d)' % (event_nr, nr_events), 'Processing: ', radar, sDatetime

            ################################################################################ 
            if event.has_key('beam_limits'):
                beamLimits_0            = event['beam_limits'][0]
                beamLimits_1            = event['beam_limits'][1]
            else:
                beamLimits_0            = default_beam_limits[0]
                beamLimits_1            = default_beam_limits[1]

            if event.has_key('gate_limits'):
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

            if new_music_obj:
                try:
                    shutil.rmtree(musicPath)
                except:
                    pass

                dataObj = msc.createMusicObj(radar.lower(), sDatetime, fDatetime
                    ,beamLimits                 = beamLimits
                    ,gateLimits                 = gateLimits
                    ,interpolationResolution    = interpRes
                    ,filterNumtaps              = numtaps 
                    ,fitfilter                  = fitfilter
                    )
            else:
                with open(picklePath,'rb') as fl:
                    dataObj     = pickle.load(fl)

            if hasattr(dataObj,'messages'):
                messages = '\n'.join([musicPath]+dataObj.messages)
                messages_filename   = os.path.join(musicPath,'messages.txt')
                with open(messages_filename,'w') as fl:
                    fl.write(messages)
                print messages
                if 'No data for this time period.' in dataObj.messages:
                    continue

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
            runParams['musicObj_path']      = picklePath
#            runParams['window_data']        = window_data
            runParams['kx_max']             = kx_max
            runParams['ky_max']             = ky_max
            runParams['autodetect_threshold'] = autodetect_threshold
            runParams['neighborhood']        = neighborhood

            runfile_obj     = msc.Runfile(radar.lower(), sDatetime, fDatetime, runParams)
            runfile_path    = runfile_obj.runParams['runfile_path']

            # Create script for viewing all plots. #########################################
            with open(os.path.join(musicPath,'0000-show_all.php'),'w') as file_obj:
                file_obj.write(show_all_txt)

            # Calculate new gatelimits and update files. ###################################
            if auto_range_on:
                try:
                    gateLimits              = auto_range(dataObj,runParams,outputDir=output_dirs['range']) 
                    runParams['gateLimits'] = gateLimits
                    runfile_obj             = msc.Runfile(radar.lower(), sDatetime, fDatetime, runParams)
                    pydarn.proc.music.defineLimits(dataObj,gateLimits=gateLimits)
                    with open(picklePath,'wb') as fl:
                        pickle.dump(dataObj,fl)
                except:
                    pass

    #            edge_detect(dataObj,runParams,outpelutDir=output_dirs['edge'])
            
            # Run MUSIC and Plotting Code ##################################################
            if only_plot_rti:
                if make_plots:
                    msc.music_plot_all(runfile_path,process_level=process_level,rti_only=True)
            else:
                msc.run_music(runfile_path,process_level=process_level)
                if make_plots:
                    msc.music_plot_all(runfile_path,process_level=process_level)

                if process_level == 'all':
                    # Print signals out to text file. ##############################################
                    with open(picklePath,'rb') as fl:
                        dataObj     = pickle.load(fl)
                    dataSets    = dataObj.get_data_sets()
                    currentData = getattr(dataObj,dataSets[-1])

                    if event.has_key('signals'):
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

                        txtPath = os.path.join(musicPath,'karr.txt')
                        with open(txtPath,'w') as fl:
                            txt = '{:<5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n'.format('Number','Kx','Ky','|K|','lambda','Azm','f','T','v','Value','Area')
                            fl.write(txt)
                            txt = '{:<5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n'.format('','[1/km]','[1/km]','[1/km]','[km]','[deg]','[mHz]','[min]','[m/s]','','[px]')
                            fl.write(txt)

                            for sigInfo in sigList:
                                txt = '{:<5}{:>10.3f}{:>10.3f}{:>10.3f}{:>10.0f}{:>10.0f}{:>10.3f}{:>10.0f}{:>10.0f}{:>10.3f}{:>10.0f}\n'.format(
                                     sigInfo['order']
                                    ,sigInfo['kx']
                                    ,sigInfo['ky']
                                    ,sigInfo['k']
                                    ,sigInfo['lambda']
                                    ,sigInfo['azm']
                                    ,sigInfo['freq'] * 1000.
                                    ,sigInfo['period']/60.
                                    ,sigInfo['vel']
                                    ,sigInfo['max']
                                    ,sigInfo['area'])
                                fl.write(txt)
        except:
            now = str(datetime.datetime.now())+':'

            err   = ' '.join(['MUSIC ERROR:',now,event['radar'],str(event['sDatetime']),str(event['fDatetime'])])
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print err
#            print '**Traceback:'
#            traceback.print_tb(exc_traceback)
#            print ''
            print '**Exception:'
            traceback.print_exc()

            with open(error_file_path,'a') as error_file:
                error_file.write(err)
                traceback.print_tb(exc_traceback,file=error_file)

if __name__ == '__main__':
    # For Case Study Paper #########################################################
#    event_list   = []
#    tmp                 = {}
#    tmp['radar']        = 'bks'
#    tmp['sDatetime']    = datetime.datetime(2010,10,19,12)
#    tmp['fDatetime']    = datetime.datetime(2010,10,20)
#    tmp['sDatetime']    = datetime.datetime(2010,10,19,14)
#    tmp['fDatetime']    = datetime.datetime(2010,10,19,16)
#    tmp['gate_limits']  = (12,30)
#    event_list.append(tmp)

    # Database Connection ########################################################## 
    sDate, eDate    = None, None
#    mstid_list, sDate, eDate = 'paper2_wal_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list, sDate, eDate = 'paper2_bks_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list, sDate, eDate = 'paper2_cvw_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list  = 'paper2_bks_2010_2011_0817mlt'
#    mstid_list  = 'paper2_wal_2012'

    mstid_lists = []
#    mstid_lists.append( 'paper2_bks_windowed_recheck')
#    mstid_list  = 'paper1_rev_bks_manual'
#    mstid_lists.append('paper2_wal')
#    mstid_lists.append('paper2_bks')
#    mstid_lists.append('paper2_fhe')
#    mstid_lists.append('paper2_fhw')
#    mstid_lists.append('paper2_cve')
#    mstid_lists.append('paper2_cvw')
#    mstid_lists.append('paper2_gbr')
#    mstid_lists.append('paper2_kap')
    mstid_lists.append('paper2_pgr')
##    mstid_lists.append('paper2_sas')
#    mstid_lists.append('automated_run_1_bks_0817mlt')
    sDate, eDate = datetime.datetime(2012,11,1),datetime.datetime(2013,5,1)

    months  = None
#    months  = [11, 12, 1]
    event_list = []
    for mstid_list in mstid_lists:
#        tmp = events_from_mongo(mstid_list,sDate=sDate,eDate=eDate,recompute=True,months=months,category='mstid')
        tmp = events_from_mongo(mstid_list,sDate=sDate,eDate=eDate,recompute=False,months=months)
        event_list += tmp

#        tmp = events_from_mongo(mstid_list,sDate=sDate,eDate=eDate,recompute=True,months=months,category='quiet')
#        event_list += tmp

#    event_list  = []
#    event_list.append({'radar':'fhw', 'sDatetime':datetime.datetime(2012,11,29,18),'fDatetime':datetime.datetime(2012,11,29,20)})

#    stime = datetime.datetime(2013,5,20,20)
#    etime = datetime.datetime(2013,5,21,3)
##    etime = stime + datetime.timedelta(hours=2)
#
#    radars = ['cvw','cve','fhw','fhe','bks','wal']
#    radars = ['pgr','sas','kap','gbr']
##    radars = ['wal']
#    event_list  = []
#    for radar in radars:
#        stimes = [stime]
#        while stimes[-1] < etime:
#            event_list.append({'radar':radar, 'sDatetime':stimes[-1],'fDatetime':stimes[-1]+datetime.timedelta(hours=2)})
#            stimes.append(stimes[-1]+datetime.timedelta(hours=2))


#    kwargs  = {}
#    kwargs['process_level'] = 'all'
#    kwargs['recompute']     = True
#    kwargs['new_music_obj'] = True 
#    kwargs['only_plot_rti'] = False
#    kwargs['auto_range_on'] = False
##    kwargs['default_gate_limits'] = (10,35)
#    kwargs['default_gate_limits'] = (12,21) #This is used for the BKS case study.
##    kwargs['default_beam_limits'] = (0,7)
#    kwargs['fitfilter']     = False
#    kwargs['firFilterLimits_0'] = 0.0005
#    kwargs['firFilterLimits_1'] = 0.0012

#    run_music(event_list[0])
    
#    for event in event_list:
#        run_music(event)

    pool = multiprocessing.Pool(4)
    pool.map(run_music,event_list)
    pool.close()
    pool.join()
