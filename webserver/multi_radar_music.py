#!/usr/bin/env python
import sys
sys.path.append('/data/mypython')

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

import utils
import pydarn

import multi_radar_music_support as msc
from auto_range import *
#from ocv_edge_detect import *
from musicRTI3 import musicRTI3
import traceback

import handling

################################################################################
def events_from_mongo(mstid_list,sDate=None,eDate=None,category=None):
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
        event_list.append(tmp)

    event_list  = sorted(event_list,key=lambda k: k['sDatetime'])
    return event_list

def check_list_for_bad_start_gate(event_list):
    #This was done because auto_range originally allowed for selection of range gates
    #too close to the radar.
    recompute_list = []
    event_nr    = 0
    for event in event_list:
#        try:
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
#        except:
#            continue

    return recompute_list

def run_music(event_list,
    recompute                   = False,
    new_music_obj               = True,
    only_plot_rti               = False,
    process_level               = 'all',
    make_plots                  = True,
    clear_output_dir            = False,
    output_dir                  = 'output',
    error_file_path             = 'static/multi_radar_music/automated_error.txt',
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
    recompute           = False #Compute event even if 014_karr.png already exists.
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
    nr_events   = len(event_list)
    event_nr    = 0
    # Run through actual events. ###################################################
    for event in event_list:
        plt.close('all')
        try:
            event_nr                += 1
            radar                   = str(event['radar'])
            radars                  = radar.split('_')
            sDatetime               = event['sDatetime']
            fDatetime               = event['fDatetime']

            ################################################################################ 

            if 'category' in event:
                print_cat = '(%s)' % event['category']
            else:
                print_cat = ''
            now = datetime.datetime.now()

            #Only run processing if we are told to not recompute and if MUSIC has been run and signals have been detected.
            picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)
            if not recompute:
                if os.path.exists(picklePath):
                    with open(picklePath,'rb') as fl:
                        test_obj    = pickle.load(fl)
                    if hasattr(test_obj.active,'sigDetect'):
                        print(now,print_cat,'(%d of %d)' % (event_nr, nr_events), radar,sDatetime,'MUSIC already computed.  Skipping!!')
                        continue

            print(now,print_cat,'(%d of %d)' % (event_nr, nr_events), 'Processing: ', radar, sDatetime)

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

            ################################################################################ 
            multi_radar_dict        = {}

            dict_keys   = []
            for dict_key in radars:
                multi_radar_dict[dict_key] = {}
                multi_radar_dict[dict_key]['rad_key'] = dict_key
                multi_radar_dict[dict_key]['merge']  = None
                dict_keys.append(dict_key)

            dict_key = radar+'-direct'
            multi_radar_dict[dict_key] = {}
            multi_radar_dict[dict_key]['rad_key'] = radar
            multi_radar_dict[dict_key]['merge']   = 'direct'
            dict_keys.append(dict_key)

            dict_key = radar+'-interp'
            multi_radar_dict[dict_key] = {}
            multi_radar_dict[dict_key]['rad_key'] = radar
            multi_radar_dict[dict_key]['merge']   = 'interp'
            dict_keys.append(dict_key)

#            for rad_key in radars+[radar]:  # Keep track of both individual and combined radars.
            for dict_key in dict_keys:
                rad_key = multi_radar_dict[dict_key]['rad_key']
                tmpd                        = multi_radar_dict[dict_key]

                tmpd['musicPath']   = msc.get_output_path(rad_key, sDatetime, fDatetime,sub_dir=tmpd['merge'])
                tmpd['picklePath']  = msc.get_pickle_name(rad_key,sDatetime,fDatetime,getPath=True,sub_dir=tmpd['merge'],createPath=False)

#                if new_music_obj:
                try:
                    shutil.rmtree(tmpd['musicPath'])
                except:
                    pass

                if rad_key != radar:    # Don't create a dataObj for the combined radars (yet).
                    tmpd['dataObj'] = msc.createMusicObj(rad_key.lower(), sDatetime, fDatetime
                        ,beamLimits                 = beamLimits
                        ,gateLimits                 = gateLimits
                        ,interpolationResolution    = interpRes
                        ,filterNumtaps              = numtaps 
                        )
#                else:
#                    tmpd['dataObj'] = pickle.load(open(tmpd['picklePath'],'rb'))

                # Create a run file. ###########################################################
                runParams = {}
                runParams['radar']              = rad_key.lower()
                runParams['sDatetime']          = sDatetime
                runParams['fDatetime']          = fDatetime
                runParams['beamLimits']         = beamLimits
                runParams['gateLimits']         = gateLimits
                runParams['interpRes']          = interpRes
                runParams['filter_numtaps']     = numtaps
                runParams['filter_cutoff_low']  = cutoff_low
                runParams['filter_cutoff_high'] = cutoff_high
                runParams['path']               = tmpd['musicPath']
                runParams['musicObj_path']      = tmpd['picklePath']
                runParams['window_data']        = window_data
                runParams['kx_max']             = kx_max
                runParams['ky_max']             = ky_max
                runParams['autodetect_threshold'] = autodetect_threshold
                runParams['neighborhood']        = neighborhood
                runParams['merge']              = tmpd['merge']

                tmpd['runfile_obj']             = msc.Runfile(rad_key.lower(), sDatetime, fDatetime, runParams,sub_dir=tmpd['merge'])
                tmpd['runfile_path']            = tmpd['runfile_obj'].runParams['runfile_path']

                # Create script for viewing all plots. #########################################
                with open(os.path.join(tmpd['musicPath'],'0000-show_all.php'),'w') as file_obj:
                    file_obj.write(show_all_txt)

            # Calculate new gatelimits and update files. ###################################
            if auto_range_on:
                gl = []
                for dict_key in dict_keys:
                    if len(dict_key.split('_')) > 1: continue
                    tmpd                    = multi_radar_dict[dict_key]
                    runParams               = tmpd['runfile_obj'].runParams
                    gl.append(auto_range(tmpd['dataObj'],runParams,outputDir=output_dirs['range']))

                gl_unzips   = list(zip(*gl))
#                gl          = np.array(gl)
                gateLimits  = (np.max(gl_unzips[0]),np.min(gl_unzips[1]))

                for dict_key in dict_keys:
                    tmpd                    = multi_radar_dict[dict_key]
                    rad_key                 = tmpd['rad_key']
                    runParams               = tmpd['runfile_obj'].runParams
                    runParams['gateLimits'] = gateLimits
                    tmpd['runfile_obj']     = msc.Runfile(rad_key.lower(), sDatetime, fDatetime, runParams)
                    if 'dataObj' in tmpd:
                        pydarn.proc.music.defineLimits(tmpd['dataObj'],gateLimits=gateLimits)
                        pickle.dump(tmpd['dataObj'],open(tmpd['picklePath'],'wb'))

                        #Plot the RTI plot for each radar.
                        msc.plot_music_rti(tmpd['dataObj'],fileName=os.path.join(tmpd['musicPath'],'000_originalFit_RTI.png'))

            if True:
                msc.run_multi_music(multi_radar_dict,radar,process_level=process_level,merge='direct')
                msc.run_multi_music(multi_radar_dict,radar,process_level=process_level,merge='interp')
                if make_plots:
                    base_merge_type = 'direct'

                    msc.merge_fan_compare(multi_radar_dict,base_merge_type=base_merge_type)

                    dict_key        = '-'.join([radar,base_merge_type])
                    tmp_output_dir  = multi_radar_dict[dict_key]['musicPath']
                    tmp_split       = os.path.split(tmp_output_dir)
                    tmp_output_dir  = tmp_split[0]
                    tmp_output_dir  = tmp_output_dir.replace(base_merge_type,'compare_pickle')  
                    handling.prepare_output_dirs({0:tmp_output_dir})

                    tmp_fName       = tmp_split[1]+'_'+radar+'.p'
                    tmp_pickle = os.path.join(tmp_output_dir,tmp_fName)
                    with open(tmp_pickle,'wb') as tp:
                        pickle.dump(multi_radar_dict,tp)
#                    msc.music_plot_all(tmpd['runfile_path'],process_level=process_level)

                if process_level == 'all':
                    # Print signals out to text file. ##############################################
#                    dataObj     = pickle.load(open(picklePath,'rb'))
                    dataObj     = tmpd['dataObj']
                    dataSets    = dataObj.get_data_sets()
                    currentData = getattr(dataObj,dataSets[-1])

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

                        txtPath = os.path.join(tmpd['musicPath'],'karr.txt')
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
            print(err)
#            print '**Traceback:'
#            traceback.print_tb(exc_traceback)
#            print ''
            print('**Exception:')
            traceback.print_exc()

            with open(error_file_path,'a') as error_file:
                error_file.write(err)
                traceback.print_tb(exc_traceback,file=error_file)
#            now = str(datetime.datetime.now())+':'
#
#            err_0   = ' '.join(['MUSIC ERROR:',now,event['radar'],str(event['sDatetime']),str(event['fDatetime'])])
#            err_1   = str(sys.exc_info())
#            
#            err     = err_0 + '\n' + err_1 + '\n'
#            print err
#
#            with open(error_file_path,'a') as error_file:
#                error_file.write(err)

if __name__ == '__main__':
    # For Case Study Paper #########################################################
    event_list   = []
    tmp                 = {}
    tmp['radar']        = 'bks'
    tmp['sDatetime']    = datetime.datetime(2010,10,19,12)
    tmp['fDatetime']    = datetime.datetime(2010,10,20)
    tmp['sDatetime']    = datetime.datetime(2010,10,19,14)
    tmp['fDatetime']    = datetime.datetime(2010,10,19,16)
    tmp['gate_limits']  = (12,30)
    event_list.append(tmp)

    # Database Connection ########################################################## 
    sDate, eDate    = None, None
#    mstid_list, sDate, eDate = 'paper2_wal_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list, sDate, eDate = 'paper2_bks_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list, sDate, eDate = 'paper2_cvw_2012',datetime.datetime(2012,11,1),datetime.datetime(2013,1,1)
#    mstid_list  = 'paper2_bks_2010_2011_0817mlt'
#    mstid_list  = 'paper2_wal_2012'
#
#    mstid_list  = 'paper2_cvw_cve'
#    event_list  = events_from_mongo(mstid_list,sDate=sDate,eDate=eDate,category='mstid')

    event_list  = []
###    event_list.append({'radar':'wal', 'sDatetime':datetime.datetime(2012,11,7,20),'fDatetime':datetime.datetime(2012,11,7,22)})
##    event_list.append({'radar':'wal', 'sDatetime':datetime.datetime(2012,12,11,20),'fDatetime':datetime.datetime(2012,12,11,22)})
##    event_list.append({'radar':'wal', 'sDatetime':datetime.datetime(2012,11,26,16),'fDatetime':datetime.datetime(2012,11,26,18)})

#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,4,10),'fDatetime':datetime.datetime(2012,11,4,12)})
#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,4,12),'fDatetime':datetime.datetime(2012,11,4,14)})
#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,4,14),'fDatetime':datetime.datetime(2012,11,4,16)})

    day = 1
    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,day,0),'fDatetime':datetime.datetime(2012,11,day,2)})
#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,day,16),'fDatetime':datetime.datetime(2012,11,day,18)})
#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,day,18),'fDatetime':datetime.datetime(2012,11,day,20)})
#    event_list.append({'radar':'fhw_fhe', 'sDatetime':datetime.datetime(2012,11,day,20),'fDatetime':datetime.datetime(2012,11,day,22)})
#
#    event_list.append({'radar':'bks_wal', 'sDatetime':datetime.datetime(2012,11,2,18),'fDatetime':datetime.datetime(2012,11,2,20)})

#    event_list  = check_list_for_bad_start_gate(event_list)

    #process_level could be 'all' or 'fft'
#    run_music(event_list,process_level='fft',recompute=True,auto_range_on=False,default_gate_limits=(16,65),only_plot_rti=False)
    run_music(event_list,process_level='fft',recompute=True,auto_range_on=True,only_plot_rti=False)
