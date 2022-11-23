#!/usr/bin/env python
#replot_rti3.py - 20 November 2014 NAF
#This routine replots RTI3 plots use for looking at MSTIDs.
#This was done because some of my old plots were not done with the correct dimensions
#and were very hard to read.

import os
import time
import datetime
import pickle
import pymongo
import music_support as msc
import multiprocessing

mongo = pymongo.MongoClient()
db    = mongo.mstid

def main(event):
    radar       = str(event['radar'])
    sDatetime   = event['sDatetime']
    eDatetime   = event['fDatetime']

    try:
        runfile_path    = msc.get_pickle_name(radar,sDatetime,eDatetime,getPath=True,runfile=True)
        runFile         = msc.load_runfile_path(runfile_path)
        musicParams     = runFile.runParams
        musicObj_path   = musicParams['musicObj_path']
        rtiPath         = os.path.join(musicParams['path'],'000_originalFit_RTI.png')
        print('http://sd-work1.ece.vt.edu/data/mstid/statistics/webserver/{0}'.format(rtiPath))

        if os.path.exists(rtiPath):
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(rtiPath))
            if mod_time > datetime.datetime(2014,3,15): return 

        dataObj         = pickle.load(open(musicObj_path,'rb'))

        xlim = (sDatetime,eDatetime)

        #Mark where sampling window starts and stops.
        dataObj.DS000_originalFit.metadata['timeLimits'] = [musicParams['sDatetime'],musicParams['fDatetime']]


        beam    = msc.get_default_beams(musicParams,dataObj)
        xlim    = msc.get_default_rti_times(musicParams,dataObj)
        ylim    = msc.get_default_gate_range(musicParams,dataObj)

        msc.plot_music_rti(dataObj,fileName=rtiPath,dataSet="originalFit",beam=beam,xlim=xlim,ylim=ylim,scale=None)
        return
    except:
        return

if __name__ == '__main__':
    mstid_list  = 'paper2_bks_2012'
    category    = 'all'

    crsr = db[mstid_list].find()

    event_list = []
    for event in crsr:
        if category != 'all':
            if 'category_manu' in event:
                if event['category_manu'] != category: continue
            elif 'category_auto' in event:
                if event['category_auto'] != category: continue

#        if event['sDatetime'] < datetime.datetime(2012,11,19,16): continue

        event_list.append(event)
    
    for event in event_list:
        main(event)

    pool = multiprocessing.Pool()
    pool.map(main,event_list)
    pool.close()
    pool.join()
