#!/usr/bin/env python
import sys
sys.path.append('/data/mypython')
import os
import shutil
import datetime
from hdf5_api import loadMusicArrayFromHDF5
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import pydarn
import multi_radar_music_support as msc
#import music_support as msc
import plot_movies

runfile_path    = '/data/mstid/statistics/webserver/static/multi_radar_music/fhw_fhe/20121103.1400-20121103.1600/fhw_fhe-20121103.1400-20121103.1600.runfile.h5'
#process_level   = 'all'
#msc.music_plot_all(runfile_path,process_level=process_level)

base_dir        = '/data/mstid/statistics/webserver/static/multi_radar_music/fhw_fhe'
dirs            = glob.glob(os.path.join(base_dir,'*'))
dirs            = []
dirs.append('/data/mstid/statistics/webserver/static/music/fhw/20121105.1400-20121105.1600')
dirs.append('/data/mstid/statistics/webserver/static/music/fhe/20121105.1400-20121105.1600')

for mydir in dirs:
    runfile_path    = glob.glob(os.path.join(mydir,'*.runfile.h5'))
#    try:
    if True:
        runfile_path    = runfile_path[0]
        print('WORKING: ', runfile_path)
        runFile         = msc.load_runfile_path(runfile_path)
        musicParams     = runFile.runParams
        musicObj_path   = musicParams['musicObj_path']
        dataObj = loadMusicArrayFromHDF5(musicObj_path)

        rad         = musicParams['radar']
        interpRes   = musicParams['interpRes']
        numtaps     = musicParams['filter_numtaps']
        cutoff_low  = musicParams['filter_cutoff_low']
        cutoff_high = musicParams['filter_cutoff_high']
        outputDir   = os.path.join(musicParams['path'],'fov_movie')
        sDatetime   = musicParams['sDatetime']
        fDatetime   = musicParams['fDatetime']
        interval    = ((musicParams['fDatetime'] - musicParams['sDatetime']).total_seconds())/2.
        half_time   = datetime.timedelta(seconds=interval)
        #time        = musicParams['sDatetime'] + half_time
        outFile     = os.path.join(outputDir,musicParams['path'].split('/')[-1]+'.mp4')

        figsize     = (20,10)
        plotSerial  = 0

        #rti_xlim    = get_default_rti_times(musicParams,dataObj)
        #rti_ylim    = get_default_gate_range(musicParams,dataObj)
        #rti_beams   = get_default_beams(musicParams,dataObj)
        #

        if os.path.exists(outputDir):
            shutil.rmtree(outputDir)
        os.makedirs(outputDir)

        plotSerial  = 0
        for time in dataObj.active.time:
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(111)
            pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,autoScale=True,time=time)
            fileName = os.path.join(outputDir,'%03i_finalDataFan.png' % plotSerial)
            fig.savefig(fileName,bbox_inches='tight')
            fig.clear()
            plt.close()
            plotSerial = plotSerial + 1
        plot_movies.make_mp4(outputDir,outFile=outFile)
#    except:
#        print 'ERROR: ',mydir
