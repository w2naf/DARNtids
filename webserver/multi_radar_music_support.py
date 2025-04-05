from flask import Flask, request, session, redirect, url_for, abort, render_template, flash, jsonify

import os
import sys
import utils
import shutil
import tempfile
import h5py
from hdf5_api import saveDictToHDF5, saveMusicArrayToHDF5, loadMusicArrayFromHDF5
import datetime
import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pymongo
from bson.objectid import ObjectId

import numpy as np
from scipy.io.idl import readsav
from scipy import signal


import pydarn
import pydarn.sdio
import pydarn.proc.music as music

from merge import merge_data

sys.path.append('/data/mypython')
import handling

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

mongo         = pymongo.MongoClient()
db            = mongo.mstid

def linkUp(dayList):
    dayStr = []
    for x in dayList:
        img1 = 'x'
        img2 = 'x'
        img3 = 'x'

        radar   = x['radar']

        musicPath = get_output_path(radar,x['sDatetime'],x['fDatetime'],create=False)
        karrCheck = glob.glob(os.path.join(musicPath,'*karr.png'))

        if 'category_manu' in x:
            img1 = 'check'

        if 'music_analysis_status' in x:
            if x['music_analysis_status']:
                img2 = 'check'

        if len(karrCheck) > 0:
            img3 = 'music_note'

        sz        = '10'
        sDate       = x['sDatetime'].strftime('%Y%m%d.%H%M') 
        fDate       = x['fDatetime'].strftime('%Y%m%d.%H%M') 
        _id         = str(x['_id'])
        url         = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id)
        xstr        = '-'.join([radar,sDate,fDate])

        anc1_tag  = '<a href="'+url+'">'
        img1_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/static/img/'+img1+'.png">'
        img2_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/static/img/'+img2+'.png">'
        img3_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/static/img/'+img3+'.png">'
        anc2_tag  = '</a>'

        fstr      = ''.join([anc1_tag,img1_tag,img2_tag,img3_tag,xstr,anc2_tag])

        dayStr.append(fstr)

    dayStr = '<br />\n'.join(dayStr)
    return dayStr

def get_prev_next(mstid_list,_id,mode='list'):
    
    if mode == 'category':
        current = db[mstid_list].find_one({'_id':_id})
        
        if 'category_manu' in current:
            category = current['category_manu']
        elif 'category_auto' in current:
            category = current['category_auto']
        else:
            category = 'None'

        items = db[mstid_list].find(
          {'$or': [
            {'category_manu': {'$exists': True},  'category_manu':category},
            {'category_manu': {'$exists': False}, 'category_auto':category}]
            }).sort([('date',1),('_id',1)])
    else:
        items = db[mstid_list].find().sort([('date',1),('_id',1)])

    inx = 0
    for item in items:
        if item['_id'] == _id:
            break
        inx += 1

    items.rewind()
    if inx == 0:
        prev_inx    = items.count()-1
    else:
        prev_inx    = inx-1

    if inx == items.count()-1:
        next_inx    = 0
    else:
        next_inx    = inx+1

    radar       = items[prev_inx]['radar']
    sDate       = items[prev_inx]['sDatetime'].strftime('%Y%m%d.%H%M') 
    fDate       = items[prev_inx]['fDatetime'].strftime('%Y%m%d.%H%M') 
    _id         = str(items[prev_inx]['_id'])
    prev_url    = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id)

    radar       = items[next_inx]['radar']
    sDate       = items[next_inx]['sDatetime'].strftime('%Y%m%d.%H%M') 
    fDate       = items[next_inx]['fDatetime'].strftime('%Y%m%d.%H%M') 
    _id         = str(items[next_inx]['_id'])
    next_url    = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id)

    return (prev_url,next_url)

def get_output_path(radar,sDatetime,fDatetime,sub_dir=None,create=True):
    lst = []
    lst.append('static')
    lst.append('multi_radar_music')
    lst.append(radar.lower())
    if sub_dir: lst.append(sub_dir)
    lst.append('-'.join([sDatetime.strftime('%Y%m%d.%H%M'),fDatetime.strftime('%Y%m%d.%H%M')]))
    path = os.path.join(*lst)
    if create:
        try:
            os.makedirs(path)
        except:
            pass
    return path

def get_hdf5_name(radar,sDatetime,fDatetime,getPath=False,sub_dir=None,createPath=False,runfile=False):
    fName = ('-'.join([radar.lower(),sDatetime.strftime('%Y%m%d.%H%M'),fDatetime.strftime('%Y%m%d.%H%M')]))+'.5'
    if getPath:
        path = get_output_path(radar,sDatetime,fDatetime,sub_dir=sub_dir,create=createPath)
        fName = os.path.join(path,fName)
        if runfile:
            fName = fName[:-2] + 'runfile.h5'

    return fName

class Runfile(object):
    def __init__(self,radar,sDatetime,fDatetime, runParamsDict,sub_dir=None):
        hdf5Path = get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,sub_dir=sub_dir,createPath=True)
        hdf5Path = hdf5Path[:-2] + 'runfile.h5'
        
        self.runParams = {}
        for key,value in runParamsDict.items():
            self.runParams[key] = value

        self.runParams['runfile_path'] = hdf5Path
        
        with h5py.File(hdf5Path, 'w') as fl:
            saveDictToHDF5(fl, self.__dict__)
        

def load_runfile_path(path):
        try:
            runFile = loadMusicArrayFromHDF5(path)
        except:
            runFile = None
        
        return runFile


def load_runfile(radar,sDatetime,fDatetime):
        hdf5Path = get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
        hdf5Path = hdf5Path[:-2] + 'runfile.h5'

        return load_runfile_path(hdf5Path)
        
def createMusicObj(radar, sDatetime, fDatetime
        ,beamLimits                 = None
        ,gateLimits                 = None
        ,interpolationResolution    = None
        ,filterNumtaps              = None
        ):

    # Calculate time limits of data needed to be loaded to make fiter work. ########
    if interpolationResolution != None and filterNumtaps != None:
        load_sDatetime,load_fDatetime = music.filterTimes(sDatetime,fDatetime,interpolationResolution,filterNumtaps)
    else:
        load_sDatetime,load_fDatetime = (sDatetime, fDatetime)

    # Load in data and create data objects. ########################################
#    myPtr   = pydarn.sdio.radDataOpen(load_sDatetime,radar,eTime=load_fDatetime,channel=channel,cp=cp,fileType=fileType,filtered=boxCarFilter)
    myPtr   = pydarn.sdio.radDataOpen(load_sDatetime,radar,eTime=load_fDatetime,filtered=True)
    dataObj = music.musicArray(myPtr,fovModel='GS')

    gl = None
    if np.size(gateLimits) == 2:
        if gateLimits[0] != None or gateLimits[1] !=None:
            if gateLimits[0] == None:
                gl0 = min(dataObj.active.fov.gates)
            else:
                gl0 = gateLimits[0]
            if gateLimits[1] == None:
                gl1 = max(dataObj.active.fov.gates)
            else:
                gl1 = gateLimits[1]
            gl = (gl0, gl1)

    if gl != None:
        music.defineLimits(dataObj,gateLimits=gl)

    bl = None
    if np.size(beamLimits) == 2:
        if beamLimits[0] != None or beamLimits[1] !=None:
            if beamLimits[0] == None:
                bl0 = min(dataObj.active.fov.beams)
            else:
                bl0 = beamLimits[0]
            if beamLimits[1] == None:
                bl1 = max(dataObj.active.fov.beams)
            else:
                bl1 = beamLimits[1]
            bl = (bl0, bl1)

    if bl != None:
        music.defineLimits(dataObj,beamLimits=bl)

    dataObj = music.checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)
    hdf5Path = get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
    saveMusicArrayToHDF5(dataObj, hdf5Path)

    return dataObj

def run_multi_music(multi_radar_dict,radar,process_level='all',merge='direct'):
    radars  = radar.split('_')

    #Generate common time vector for interpolation.
    tmp_times       = []
    tmp_interpRes   = []
    for rad_key in radars:
        tmpd            = multi_radar_dict[rad_key]
        tmp_times.append(tmpd['dataObj'].active.time.min())
        tmp_times.append(tmpd['dataObj'].active.time.max())
        tmp_interpRes.append(tmpd['runfile_obj'].runParams['interpRes'])

    timeLim     = (min(tmp_times),max(tmp_times))
    interpRes   = min(tmp_interpRes)

    timeVec     = [min(timeLim)]
    curr_time   = min(timeLim)
    while True:
        curr_time += datetime.timedelta(seconds=interpRes)
        if curr_time > timeLim[1]:
            break

        timeVec.append(curr_time)

    for rad_key in radars:
        tmpd            = multi_radar_dict[rad_key]

#        runFile         = load_runfile_path(runfile_path)
        runFile         = tmpd['runfile_obj']
        musicParams     = runFile.runParams
        musicObj_path   = tmpd['hdf5Path']
        dataObj         = tmpd['dataObj']

        rad         = musicParams['radar']
        sDatetime   = musicParams['sDatetime']
        fDatetime   = musicParams['fDatetime']
        numtaps     = musicParams['filter_numtaps']
        cutoff_low  = musicParams['filter_cutoff_low']
        cutoff_high = musicParams['filter_cutoff_high']
        kx_max      = musicParams['kx_max']
        ky_max      = musicParams['ky_max']
        threshold   = musicParams['autodetect_threshold']
        neighborhood = musicParams['neighborhood']

        dataObj = music.checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)
        if not dataObj.active.metadata['good_period']:
            saveMusicArrayToHDF5(dataObj, musicObj_path)
            return

        dataObj.active.applyLimits()
        music.beamInterpolation(dataObj,dataSet='limitsApplied')


        music.timeInterpolation(dataObj,newTimeVec=timeVec)
        saveMusicArrayToHDF5(dataObj, musicObj_path)

    dataObj_0   = multi_radar_dict[radars[0]]['dataObj']
    dataObj_1   = multi_radar_dict[radars[1]]['dataObj']

    # Compute the direct, concatenated version of the merge.
    if merge == 'direct': interp=False
    else: interp = True
    dataObj     = merge_data(dataObj_0,dataObj_1,interp=interp)

#    # Also compute the interpolated version of the merge and save it into a
#    # different data set name.  Whichever is the one set as active is the one 
#    # that gets plotted.
#    tmp_interp      = (merge.merge_data(dataObj_0,dataObj_1,interp=True)).active
#    merged_interp   = dataObj.active.copy('merged_interp','Merged Interp')
#    merged_interp.data              = tmp_interp.data            
#    merged_interp.time              = tmp_interp.time
#    merged_interp.fov.gates         = tmp_interp.fov.gates       
#    merged_interp.fov.beams         = tmp_interp.fov.beams       
#    merged_interp.fov.latCenter     = tmp_interp.fov.latCenter   
#    merged_interp.fov.latFull       = tmp_interp.fov.latFull     
#    merged_interp.fov.lonCenter     = tmp_interp.fov.lonCenter   
#    merged_interp.fov.lonFull       = tmp_interp.fov.lonFull     
#    merged_interp.fov.slantRCenter  = tmp_interp.fov.slantRCenter
#    merged_interp.fov.slantRFull    = tmp_interp.fov.slantRFull  
#    merged_interp.setActive()

    multi_radar_dict[radar+'-'+merge]['dataObj'] = dataObj

    music.determineRelativePosition(dataObj)

    tmpd            = multi_radar_dict[radar+'-'+merge]
    runFile         = tmpd['runfile_obj']
    musicParams     = runFile.runParams
    musicObj_path   = tmpd['hdf5Path']
    dataObj         = tmpd['dataObj']

    rad         = musicParams['radar']
    sDatetime   = musicParams['sDatetime']
    fDatetime   = musicParams['fDatetime']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    kx_max      = musicParams['kx_max']
    ky_max      = musicParams['ky_max']
    threshold   = musicParams['autodetect_threshold']
    neighborhood = musicParams['neighborhood']

    music.nan_to_num(dataObj)

    filt        = music.filter(dataObj, dataSet='active', numtaps=numtaps, cutoff_low=cutoff_low, cutoff_high=cutoff_high)

    outputDir   = musicParams['path']
    figsize     = (20,10)
    plotSerial  = 999
    fig = plt.figure(figsize=figsize)
    filt.plotImpulseResponse(fig=fig)
    fileName = os.path.join(outputDir,'%03i_impulseResponse.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
#    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    filt.plotTransferFunction(fig=fig,xmax=0.004)
    fileName = os.path.join(outputDir,'%03i_transferFunction.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
#    plotSerial = plotSerial + 1

    music.detrend(dataObj, dataSet='active')

    if musicParams['window_data']:
        music.windowData(dataObj, dataSet='active')

    music.calculateFFT(dataObj)
    if process_level == 'fft':
        saveMusicArrayToHDF5(dataObj, musicObj_path)
        return

    music.calculateDlm(dataObj)
    music.calculateKarr(dataObj,kxMax=kx_max,kyMax=ky_max)
    music.detectSignals(dataObj,threshold=threshold,neighborhood=neighborhood)

    saveMusicArrayToHDF5(dataObj, musicObj_path)

def music_plot_all(runfile_path,process_level='all'):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj = loadMusicArrayFromHDF5(musicObj_path)

    rad         = musicParams['radar']
    interpRes   = musicParams['interpRes']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    outputDir   = musicParams['path']
    sDatetime   = musicParams['sDatetime']
    fDatetime   = musicParams['fDatetime']
    interval    = ((musicParams['fDatetime'] - musicParams['sDatetime']).total_seconds())/2.
    half_time   = datetime.timedelta(seconds=interval)
    time        = musicParams['sDatetime'] + half_time

    figsize     = (20,10)
    plotSerial  = 0

    rti_xlim    = get_default_rti_times(musicParams,dataObj)
    rti_ylim    = get_default_gate_range(musicParams,dataObj)
    rti_beams   = get_default_beams(musicParams,dataObj)

    dataObj.DS000_originalFit.metadata['timeLimits'] = [musicParams['sDatetime'],musicParams['fDatetime']]
    fileName = os.path.join(outputDir,'%03i_originalFit_RTI.png' % plotSerial)
    plot_music_rti(dataObj,
            fileName    = fileName,
            dataSet     = "originalFit",
            beam        = rti_beams,
            xlim        = rti_xlim,
            ylim        = rti_ylim)

    dataObj.DS000_originalFit.metadata.pop('timeLimits',None)
    plotSerial = plotSerial + 1

    if 'good_period' in dataObj.active.metadata:
        if not dataObj.active.metadata['good_period']:
            return

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(121)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,dataSet='originalFit',axis=ax,time=time)
    ax  = fig.add_subplot(122)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,dataSet='beamInterpolated',time=time)
    fileName = os.path.join(outputDir,'%03i_beamInterp_fan.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    try:
        fig = plt.figure(figsize=figsize)
        pydarn.plotting.musicPlot.plotRelativeRanges(dataObj,time=time,fig=fig,dataSet='beamInterpolated')
        fileName = os.path.join(outputDir,'%03i_ranges.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        fig.clear()
        plt.close()
    except:
        pass
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,dataSet="DS002_beamInterpolated",dataSet2='DS001_limitsApplied',fig=fig)
    fileName = os.path.join(outputDir,'%03i_beamInterp.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,dataSet='timeInterpolated',dataSet2='beamInterpolated',fig=fig)
    fileName = os.path.join(outputDir,'%03i_timeInterp.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig,dataSet="DS005_filtered")
    fileName = os.path.join(outputDir,'%03i_filtered.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig)
    fileName = os.path.join(outputDir,'%03i_detrendedData.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    if musicParams['window_data']:
        fig = plt.figure(figsize=figsize)
        pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig)
        fileName = os.path.join(outputDir,'%03i_windowedData.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        fig.clear()
        plt.close()
        plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,xlim=(-0.0025,0.0025))
    fileName = os.path.join(outputDir,'%03i_spectrum.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,plotType='magnitude',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_magnitude.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,plotType='phase',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_phase.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,autoScale=True,time=time)
    fileName = os.path.join(outputDir,'%03i_finalDataFan.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicRTI(dataObj,plotZeros=True,axis=ax,autoScale=True)
    fileName = os.path.join(outputDir,'%03i_finalDataRTI.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.plotFullSpectrum(dataObj,fig=fig,xlim=(0,0.0015))
    fileName = os.path.join(outputDir,'%03i_fullSpectrum.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    if process_level == 'fft': return

    fig = plt.figure(figsize=figsize)
    pydarn.plotting.musicPlot.plotDlm(dataObj,fig=fig)
    fileName = os.path.join(outputDir,'%03i_dlm_abs.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=(10,10))
    pydarn.plotting.musicPlot.plotKarr(dataObj,fig=fig,maxSignals=25)
    fileName = os.path.join(outputDir,'%03i_karr.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()
    plotSerial = plotSerial + 1

#    fig = plt.figure(figsize=(10,10))
#    pydarn.plotting.musicPlot.plotKarrDetected(dataObj,fig=fig)
#    fileName = os.path.join(outputDir,'%03i_karrDetected.png' % plotSerial)
#    fig.savefig(fileName,bbox_inches='tight')
#    fig.clear()
#    plt.close()
#    plotSerial = plotSerial + 1

def music_plot_fan(runfile_path,time=None,fileName='fan.png',scale=None):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = loadMusicArrayFromHDF5(musicObj_path)

    rad         = musicParams['radar']
    interpRes   = musicParams['interpRes']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    outputDir   = musicParams['path']

    figsize     = (20,10)
    plotSerial  = 0

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(121)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,dataSet='originalFit',axis=ax,time=time,scale=scale)
    ax  = fig.add_subplot(122)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,dataSet='beamInterpolated',time=time,scale=scale)
    fullPath = os.path.join(outputDir,fileName)
    fig.savefig(fullPath,bbox_inches='tight')
    fig.clear()
    plt.close()

class MultiRadarMapPlot(object):
    def __init__(self,
            width           = 10000000,
            height          = 10000000,
            lon_0           = -100,
            lat_0           = 40,
            time            = None,
            scale           = None,
            cmap_handling   = 'superdarn',
            autoScale       = False,
            alpha           = 1,
            cbar_ticks      = None,
            cbar_shrink             = 1.0,
            cbar_fraction           = 0.15,
            cbar_gstext_offset      = -0.075,
            cbar_gstext_fontsize    = None,
            plot_cbar       = True,
            basemap_dict    = {},
            model_text_size = 'small',
            ax              = None):

        from mpl_toolkits.basemap import Basemap

        if ax is None:
            fig = plt.figure(figsize=(10,8))
            ax  = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

#        m = Basemap(projection='stere',width=width,height=height,lon_0=lon_0,lat_0=lat_0,ax=ax,**basemap_dict)
        m = utils.plotUtils.mapObj(projection='stere',datetime=time,width=width,height=height,lon_0=lon_0,lat_0=lat_0,ax=ax,**basemap_dict)

        parallels_ticks = np.arange(-80.,81.,10.)
        meridians_ticks = np.arange(-180.,181.,20.)

        m.drawparallels(parallels_ticks,labels=[1,0,0,0])
        m.drawmeridians(meridians_ticks,labels=[0,0,0,1])

        m.drawcoastlines(linewidth=0.5,color='k')
        m.drawmapboundary(fill_color='w')
        m.fillcontinents(color='w', lake_color='w')

        self.fig                = fig
        self.ax                 = ax
        self.m                  = m 
        self.time               = time
        self.verts              = []
        self.vals               = []
        self.scale              = scale
        self.datasets           = []
        self.prms               = []
        self.tfreqs             = []
        self.cmap_handling      = cmap_handling
        self.autoScale          = autoScale
        self.cbar_ticks         = cbar_ticks
        self.plot_cbar          = plot_cbar
        self.model_text_size    = model_text_size
        self.data_titles        = []
        self.alpha              = alpha

        self.cbar_shrink             = cbar_shrink
        self.cbar_fraction           = cbar_fraction
        self.cbar_gstext_offset      = cbar_gstext_offset
        self.cbar_gstext_fontsize    = cbar_gstext_fontsize

    def add_data(self,currentData,prm=None):
        self.datasets.append(currentData)
        self.prms.append(prm)

        m       = self.m
        time    = self.time
        #Figure out which scan we are going to plot...
        if time == None:
            timeInx = 0
            self.time   = currentData.time[timeInx]
        else:
            timeInx = (np.where(currentData.time >= time))[0]
            if np.size(timeInx) == 0:
                timeInx = -1
            else:
                timeInx = int(np.min(timeInx))


        latFull     = currentData.fov.latFull
        lonFull     = currentData.fov.lonFull

        lonFull,latFull = (np.array(lonFull)+360.)%360.,np.array(latFull)

        goodLatLon  = np.logical_and( np.logical_not(np.isnan(lonFull)), np.logical_not(np.isnan(latFull)) )
        goodInx     = np.where(goodLatLon)
        goodLatFull = latFull[goodInx]
        goodLonFull = lonFull[goodInx]

        #Plot the SuperDARN data!
        ngates  = np.shape(currentData.data)[2]
        nbeams  = np.shape(currentData.data)[1]
        data    = currentData.data[timeInx,:,:]
        for bm in range(nbeams):
            for rg in range(ngates):
                if goodLatLon[bm,rg] == False: continue
                if np.isnan(data[bm,rg]): continue
                if data[bm,rg] == 0 and not plotZeros: continue
                self.vals.append(data[bm,rg])

                x1,y1 = m(lonFull[bm+0,rg+0],latFull[bm+0,rg+0])
                x2,y2 = m(lonFull[bm+1,rg+0],latFull[bm+1,rg+0])
                x3,y3 = m(lonFull[bm+1,rg+1],latFull[bm+1,rg+1])
                x4,y4 = m(lonFull[bm+0,rg+1],latFull[bm+0,rg+1])
                self.verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

        md      = currentData.metadata
        title   = '{radar} {dataset}: {time}'.format(
                radar   = md['code'][0].upper(),
                dataset = md['dataSetName'],
                time    = self.time.strftime('%d %b %Y %H%M UT'))

        #Get the frequency
        if prm is not None:
            prm_times   = np.array(prm.time)
            tmp_inxs    = np.where(np.array(prm.time) <= time)[0]

            tmp_times   = prm_times[tmp_inxs]
            tmp_tfreqs  = np.array(prm.tfreq)[tmp_inxs]

            inx         = tmp_times.argmax()
            prm_time    = tmp_times[inx]
            prm_tfreq   = tmp_tfreqs[inx]

            self.tfreqs.append({'radar':md['code'][0].upper(), 'tfreq':prm_tfreq, 'time':prm_time})

            title = '{0} - {1:.1f} MHz'.format(title,prm_tfreq/1000.) 

        self.data_titles.append(title)

    def plot(self):
        from matplotlib.collections import PolyCollection
        from pydarn.radar.radUtils import getParamDict
        import scipy.stats as stats

        #Set colorbar scale if not explicitly defined.

        ################################################################################
        #Check to make sure we are actually plotting the same thing in all data sets!!
        md_params   = []
        for dataset in self.datasets:
            md_params.append(dataset.metadata['param'])

        if np.unique(np.array(md_params)).size != 1:
            print('WARNING!  Not all data sets are the same parameter!!')
            return
        else:
            md_param = md_params[0]

        models  = []
        for dataset in self.datasets:
            models.append(dataset.metadata['model'])

        if np.unique(np.array(models)).size != 1:
            print('WARNING!  Not all data sets are the same geolocation model!!')
            return
        else:
            model = models[0]

        coords  = []
        for dataset in self.datasets:
            coords.append(dataset.metadata['coords'])

        if np.unique(np.array(coords)).size != 1:
            print('WARNING!  Not all data sets are the same coordinate system!!')
            return
        else:
            coord = coords[0]

        gscats  = []
        for dataset in self.datasets:
            gscats.append(dataset.metadata.get('gscat'))
        if np.unique(np.array(gscats)).size != 1:
            print('WARNING!  Not all data sets are the same ground scatter flag!!')
            return
        else:
            gscat = gscats[0]
        ################################################################################

        #Translate parameter information from short to long form.
        paramDict = getParamDict(md_param)
        if 'label' in paramDict:
            param     = paramDict['param']
            cbarLabel = paramDict['label']
        else:
            param       = 'width' #Set param = 'width' at this point just to not screw up the colorbar function.
            cbarLabel   = param

        #Set colorbar scale if not explicitly defined.
        if(self.scale == None):
            if self.autoScale:
                sd          = stats.nanstd(np.abs(self.vals),axis=None)
                mean        = stats.nanmean(np.abs(self.vals),axis=None)
                scMax       = np.ceil(mean + 1.*sd)
                if np.nanmin(self.vals) < 0:
                    self.scale   = scMax*np.array([-1.,1.])
                else:
                    self.scale   = scMax*np.array([0.,1.])
            else:
                if 'range' in paramDict:
                    self.scale = paramDict['range']
                else:
                    self.scale = [-200,200]

        if (self.cmap_handling == 'matplotlib') or self.autoScale:
            cmap    = matplotlib.cm.jet
            bounds  = np.linspace(self.scale[0],self.scale[1],256)
            norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
        elif self.cmap_handling == 'superdarn':
            cmap,norm,bounds = utils.plotUtils.genCmap(param,self.scale,colors='lasse')

        #Plot data on map.
        pcoll = PolyCollection(np.array(self.verts),edgecolors='face',closed=False,
                cmap=cmap,norm=norm,zorder=99,alpha=self.alpha)
        pcoll.set_array(np.array(self.vals))
        self.ax.add_collection(pcoll,autolim=False)

        txt = 'Coordinates: ' + coord +', Model: ' + model
        self.ax.text(1.01, 0, txt,
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  rotation='vertical',
                  size=self.model_text_size,
                  transform=self.ax.transAxes)

        if self.plot_cbar:
            cbar = self.fig.colorbar(pcoll,orientation='vertical',shrink=self.cbar_shrink,fraction=self.cbar_fraction)
            cbar.set_label(cbarLabel)
            if self.cbar_ticks is None:
                labels = cbar.ax.get_yticklabels()
                labels[-1].set_visible(False)
            else:
                cbar.set_ticks(self.cbar_ticks)

            if gscat == 1:
                cbar.ax.text(0.5,self.cbar_gstext_offset,'Ground\nscat\nonly',ha='center',fontsize=self.cbar_gstext_fontsize)

        self.m.nightshade(self.time)

        #FHW: 204, FHE: 205
        for ds in self.datasets:
            stid    = ds.metadata['stid']
            pydarn.plotting.overlayFov(self.m,ids=stid,dateTime=self.time,model=ds.metadata['model'])

        tfreq_text  = []
        tfreqs      = []
        for tfreq_dict in self.tfreqs:
            txt = '{0} {1} UT {2:d} kHz'.format(tfreq_dict['radar'], tfreq_dict['time'].strftime('%H%M'), tfreq_dict['tfreq'])
            tfreq_text.append(txt)
            tfreqs.append(tfreq_dict['tfreq'])

        tfreqs      = np.array(tfreqs) 
        tfreq_diff  = tfreqs.max() - tfreqs.min()
        if tfreq_diff > 500:
            tfreq_color = 'red'
        else:
            tfreq_color = 'green'

        self.ax.text(0.01,0.01,'\n'.join(tfreq_text),transform=self.ax.transAxes,fontdict={'family':'monospace','weight':'bold','size':'x-large'},color=tfreq_color)

        self.ax.set_title('\n'.join(self.data_titles))

        return

def merge_fan_compare(multi_radar_dict,time=None,fileName=None,full_path=None,scale=None,alpha=0.60,
        panels=['raw','direct','interp','filtered-direct'],
        base_merge_type='direct'):

    #base_merge_type = 'direct' # We have to choose 1 type that we are going to pull times, output dirs from.

    # Figure out where the plots are going.
    for radar,radar_dict in multi_radar_dict.items():
        split_0 = radar.split('_')
        if len(split_0) == 2:
            split_1 = split_0[1].split('-')
            if split_1[1] == base_merge_type:
                output_dir  = os.path.split(radar_dict['musicPath'])[0]
                rad_key     = radar_dict['rad_key']
                if time is None:
                    for dataset_name in radar_dict['dataObj'].get_data_sets():
                        if 'merged_{}'.format(base_merge_type) in dataset_name:
                            current_data = getattr(radar_dict['dataObj'],dataset_name)
                            time = min(current_data.time)

    if full_path is None:
        output_dir  = output_dir.replace(base_merge_type,'compare_png_singles')
        handling.prepare_output_dirs({0:output_dir})
        if fileName is None:
            fileName = time.strftime('%Y%m%d_%H%M_compare.png')

        full_path       = os.path.join(output_dir,fileName)
    else:
        output_dir, fileName = os.path.split(full_path)

    # Setup Figure
    if len(panels) > 1:
        nr_subplots_x   = 2
    else:
        nr_subplots_x   = 1

    if len(panels) > 2:
        nr_subplots_y   = 2
    else:
        nr_subplots_y   = 1

    subplot_width   = 10
    subplot_height  = 6 
    figsize         = (nr_subplots_x*subplot_width,nr_subplots_y*subplot_height)
    fig             = plt.figure(figsize=figsize)
    
    #Good set of map params for FHW/FHE
    lat_0       =   45
    lon_0       = -97.5
    map_width   = 2800000
    map_height  = 1700000

#    lat_0       =  50
#    lon_0       = -85.0
#    map_width   = 4500000
#    map_height  = 4500000

#    lat_0       =  90
#    lon_0       = -85.0
#    map_width   = 45000000
#    map_height  = 45000000

    cbar_shrink         =  0.65
    cbar_gstext_offset  = -0.150

    plot_nr     = 0
    for panel in panels:
        #aa.DS000_originalFit       aa.DS003_timeInterpolated  aa.DS006_nan_to_num       aa.DS009_detrended         aa.messages
        #aa.DS001_limitsApplied     aa.DS004_merged_direct     aa.DS007_filtered          aa.active                  aa.prm
        #aa.DS002_beamInterpolated  aa.DS005_merged_interp     aa.DS008_limitsApplied     aa.get_data_sets 
        plot_nr += 1
        ax = fig.add_subplot(nr_subplots_y,nr_subplots_x,plot_nr)

        if panel == 'raw':
            _scale = scale
            mm = MultiRadarMapPlot(scale=_scale,cbar_ticks=None,width=map_width,height=map_height,
                    lat_0=lat_0,lon_0=lon_0,ax=ax,time=time,
                    cbar_shrink=cbar_shrink,cbar_gstext_offset=cbar_gstext_offset,alpha=alpha)
            for radar,radar_dict in multi_radar_dict.items():
                if len(radar.split('_')) == 1:
                    dataObj = radar_dict['dataObj']
                    mm.add_data(dataObj.DS000_originalFit,prm=dataObj.prm)
            mm.plot()
            continue

        if (panel == 'direct') or (panel == 'interp'):
            dataObj         = multi_radar_dict['-'.join([rad_key,panel])]['dataObj']
            dataSet         = 'merged_{}'.format(panel)
            _scale          = scale
            _cbar_ticks     = None
            _cmap_handling  = 'superdarn'

        if (panel == 'filtered-interp') or (panel == 'filtered-direct'):
            merge_type      = panel.split('-')[1]
            dataObj         = multi_radar_dict['-'.join([rad_key,merge_type])]['dataObj']
            dataSet         = 'limitsApplied'
            _scale          = (-2,2)
            _cbar_ticks      = [-2.0, -1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5, 2.0]
            _cmap_handling  = 'matplotlib'

        pydarn.plotting.musicPlot.musicFan(dataObj,dataSet=dataSet,time=time,time_tolerance=datetime.timedelta(minutes=2),
                width=map_width,height=map_height,lat_0=lat_0,lon_0=lon_0,
                scale=_scale, plotZeros=True, alpha=alpha,
                cbar_shrink=cbar_shrink,cbar_ticks=_cbar_ticks,cmap_handling=_cmap_handling,cbar_gstext_offset=cbar_gstext_offset,
                max_title_len=70,axis=ax)

    fig.tight_layout(h_pad=-1.,w_pad=0)
    fig.savefig(full_path,bbox_inches='tight')
    plt.close(fig)

def music_plot_karr(runfile_path,fileName='karr.png',maxSignals=25):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = loadMusicArrayFromHDF5(musicObj_path)

    fig = plt.figure(figsize=(10,10))
    pydarn.plotting.musicPlot.plotKarr(dataObj,fig=fig,maxSignals=maxSignals)
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()

def plot_music_rti(dataObj
        , dataSet='active'
        , beam=[4,7,13]
        , xlim=None
        , ylim=None
        , coords='gate'
        , fileName='rti.png'
        , scale=None
        , plotZeros=False
        , xBoundaryLimits=None
        , yBoundaryLimits=None
        , autoScale=False
        , axvlines = None
        , figsize = (20,15)
        ):

    from musicRTI3 import musicRTI3

    fig     = plt.figure(figsize=figsize)
    axis    = fig.add_subplot(111)
#    pydarn.plotting.musicPlot.musicRTI(dataObj
    musicRTI3(dataObj
        , dataSet=dataSet
        , beams=beam
        , xlim=xlim
        , ylim=ylim
        , coords=coords
        , axis=axis
        , scale=scale
        , plotZeros=plotZeros
        , xBoundaryLimits=xBoundaryLimits
        , yBoundaryLimits=yBoundaryLimits
        , axvlines = axvlines
        , autoScale=autoScale
        )
    fig.savefig(fileName,bbox_inches='tight')
    fig.clear()
    plt.close()

def get_default_rti_times(musicParams,dataObj=None,min_hours=8):
    #Set up suggested boundaries for RTI replotting.
    min_timedelta = datetime.timedelta(hours=min_hours)
    duration = musicParams['fDatetime'] - musicParams['sDatetime']
    if duration < min_timedelta:
        center_time = musicParams['sDatetime'] + duration/2
        min_time = center_time - min_timedelta/2
        max_time = center_time + min_timedelta/2
    else:
        min_time = musicParams['sDatetime']
        max_time = musicParams['fDatetime']

    return min_time,max_time

def get_default_gate_range(musicParams,dataObj=None,gate_buffer=10):
    min_gate = None
    max_gate = None

    if 'gateLimits' in musicParams:
        if musicParams['gateLimits'] is not None:
            if musicParams['gateLimits'][0] is not None:
                min_gate = musicParams['gateLimits'][0] - gate_buffer
                if dataObj is not None:
                    gts = dataObj.DS000_originalFit.fov.gates
                    if min_gate < min(gts): min_gate = min(gts)
            if musicParams['gateLimits'][1] is not None:
                max_gate = musicParams['gateLimits'][1] + gate_buffer
                if dataObj is not None:
                    gts = dataObj.DS000_originalFit.fov.gates
                    if max_gate > max(gts): max_gate = max(gts)

    return min_gate,max_gate

def get_default_beams(musicParams,dataObj=None,beams=[4,7,13]):
    if dataObj is not None:
        new_beam_list = []
        bms = dataObj.DS000_originalFit.fov.beams
        for beam in beams:
            if beam in bms:
                new_beam_list.append(beam)
            else:
                new_beam_list.append(bms[0])
    else:
        new_beam_list = beams
    return new_beam_list
