# from flask import Flask, request, session, redirect, url_for, abort, render_template, flash, jsonify

import os
import sys
# from davitpy import utils
# import shutil
import tempfile
from hdf5_api import loadMusicArrayFromHDF5, saveMusicArrayToHDF5, saveDictToHDF5
import h5py
import datetime
import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pymongo
# from bson.objectid import ObjectId

import numpy as np
# from scipy.io.idl import readsav
# from scipy import signal


from pyDARNmusic import (load_fitacf,getDataSet,musicFan,musicRTP,music ,stringify_signal,stringify_signal_list     
                               ,beamInterpolation         
                               ,defineLimits              
                               ,checkDataQuality          
                               ,applyLimits               
                               ,determineRelativePosition 
                               ,timeInterpolation         
                               ,filterTimes               
                               ,detrend                   
                               ,nan_to_num                
                               ,windowData                
                               ,calculateFFT              
                               ,calculateDlm              
                               ,calculateKarr             
                               ,simulator                 
                               ,scale_karr                
                               ,detectSignals             
                               ,add_signal                
                               ,del_signal,
                               timeSeriesMultiPlot
                                 ,plotRelativeRanges
                                 ,spectrumMultiPlot
                                 ,plotFullSpectrum
                                 ,plotDlm
                                 ,plotKarr
                                 ,plotKarrDetected,daynight_terminator)
# import davitpy.pydarn.proc.music as music

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

mongo         = pymongo.MongoClient()
db            = mongo.mstid

def linkUp(dayList):
    dayStr = []
    for x in dayList:
        img1 = 'x'
        img2 = 'x'
        img3 = 'x'

        title1 = 'category_manu not set.'
        title2 = 'Event is marked incomplete.'
        title3 = 'dataObj file does not exist.'

        radar   = x['radar']

        musicPath   = get_output_path(radar,x['sDatetime'],x['fDatetime'],create=False)
        dataObjPath = get_hdf5_name(radar,x['sDatetime'],x['fDatetime'],getPath=True)
        karrCheck   = glob.glob(os.path.join(musicPath,'*karr.png'))

        if 'category_manu' in x:
            img1 = 'check'
            title1 = 'Item has been classified.'

        if 'music_analysis_status' in x:
            if x['music_analysis_status']:
                img2 = 'check'
                title2 = 'Event has been marked as complete.'
        
        if os.path.exists(dataObjPath):
            img3 = 'music_note_gray'
            title3 = 'dataObj file has been created.'

        if len(karrCheck) > 0:
            img3 = 'music_note'
            title3 = 'karr has been plotted.'

        sz        = '10'
        sDate       = x['sDatetime'].strftime('%Y%m%d.%H%M') 
        fDate       = x['fDatetime'].strftime('%Y%m%d.%H%M') 
        _id         = str(x['_id'])
        # url         = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id) 
        url = "music_edit/"+radar+"/"+sDate+"/"+fDate+"/"+_id
        xstr        = '-'.join([radar,sDate,fDate])
        # onclick="getMusicEdit(\''+radar+'\',\''+sDate+'\',\''+fDate+'\',\''+_id+'\');"
        base = "apphome"

        anc1_tag  = '<a href="/music_edit/?radar='+radar+'&sDate='+sDate+'&fDate='+fDate+'&id='+_id+'" class="btn btn-primary mt-2">'
        # anc1_tag  = '<a href="'+url+'">'
        img1_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/staticfiles/images/'+img1+'.png" title="'+title1+'">'
        img2_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/staticfiles/images/'+img2+'.png" title="'+title2+'">'
        img3_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/staticfiles/images/'+img3+'.png" title="'+title3+'">'
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
            items = db[mstid_list].find({'category_manu':category}).sort([('date',1),('_id',1)])
        else:
            items = db[mstid_list].find({'category_manu': {'$exists': False}}).sort([('date',1),('_id',1)])
    else:
        items = db[mstid_list].find().sort([('date',1),('_id',1)])

    inx = 0
    for item in items:
        if item['_id'] == _id:
            break
        inx += 1
    # inx = 2
    items.rewind()
    count = 0
    count = len(list(items))
    if inx == 0:
        prev_inx    = count-1
    else:
        prev_inx    = inx-1

    if inx == count-1:
        next_inx    = 0
    else:
        next_inx    = inx+1
    
    items.rewind()
    radar       = items[prev_inx]['radar']
    sDate       = items[prev_inx]['sDatetime'].strftime('%Y%m%d.%H%M') 
    fDate       = items[prev_inx]['fDatetime'].strftime('%Y%m%d.%H%M') 
    _id         = str(items[prev_inx]['_id'])
    # prev_url    = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id)
    # prev_url = "music_edit/"+radar+"/"+sDate+"/"+fDate+"/"+_id
    prev_url = "/music_edit/?radar="+radar+"&sDate="+sDate+"&fDate="+fDate+"&id="+_id
    radar       = items[next_inx]['radar']
    sDate       = items[next_inx]['sDatetime'].strftime('%Y%m%d.%H%M') 
    fDate       = items[next_inx]['fDatetime'].strftime('%Y%m%d.%H%M') 
    _id         = str(items[next_inx]['_id'])
    # next_url    = url_for('music_edit',radar=radar,sDate=sDate,fDate=fDate,_id=_id)
    # next_url = "music_edit/"+radar+"/"+sDate+"/"+fDate+"/"+_id
    next_url = "/music_edit/?radar="+radar+"&sDate="+sDate+"&fDate="+fDate+"&id="+_id
    return (prev_url,next_url)

def get_enabled_sources(path='staticfiles/music_sources_enabled'):
    files = glob.glob(os.path.join(path,'*'))

    result_list = []
    for fl in files:
        result_list.append( (os.path.basename(fl), os.path.realpath(fl)) )

    return result_list

def sourcesDropDown(**kwargs):
    current_source  = get_output_path(real_path=True)
    enabled_sources = get_enabled_sources(**kwargs)

    html = []
    html.append('<select name="sources_dropdown" id="sources_dropdown">')
    html.append('   <option value="current" selected>Current ({})</option>'.format(current_source))
    html.append('   <option value="current" disabled>----------</option>')
    for x in enabled_sources:
        value = x[1]
        text  = '{} ({})'.format(x[0],x[1])
        tag = '  <option value="{}">{}</option>'.format(value,text)
        html.append(tag)

    html.append('</select>')
    html.append('<button disabled id="sourceSelectorButton">Select Source</button>')
    html = '\n'.join(html)
    return html

def get_output_path(radar=None,sDatetime=None,fDatetime=None,data_path='staticfiles/music',create=False,real_path=False):
    if real_path:
        data_path = os.path.realpath(data_path)

    if radar is None and sDatetime is None and fDatetime is None:
        return data_path

    lst = []
    lst.append(data_path)
    lst.append(radar.lower())
    lst.append('-'.join([sDatetime.strftime('%Y%m%d.%H%M'),fDatetime.strftime('%Y%m%d.%H%M')]))
    path = os.path.join(*lst)
    if create:
        try:
            os.makedirs(path)
        except:
            pass
    return path

def get_hdf5_name(radar,sDatetime,fDatetime,getPath=False,createPath=False,runfile=False,**kwargs):
    fName = ('-'.join([radar.lower(), sDatetime.strftime('%Y%m%d.%H%M'), fDatetime.strftime('%Y%m%d.%H%M')])) + '.h5'
    if getPath:
        path    = get_output_path(radar,sDatetime,fDatetime,create=createPath,**kwargs)
        fName   = os.path.join(path,fName)
        if runfile:
            fName = fName[:-2] + 'runfile.h5'

    return fName

class Runfile(object):
    def __init__(self,radar,sDatetime,fDatetime, runParamsDict):
        hdf5Path = get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
        hdf5Path = hdf5Path[:-2] + 'runfile.h5'
        
        self.runParams = {}
        for key,value in runParamsDict.items():
            self.runParams[key] = value
  
        self.runParams['runfile_path'] = hdf5Path
        
        
        with h5py.File(runfile_path, 'w') as fl:
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
        ,fitfilter                  = False
        ):

    # Calculate time limits of data needed to be loaded to make fiter work. ########
    if interpolationResolution != None and filterNumtaps != None:
        load_sDatetime,load_fDatetime = filterTimes(sDatetime,fDatetime,interpolationResolution,filterNumtaps)
    else:
        load_sDatetime,load_fDatetime = (sDatetime, fDatetime)

    # Load in data and create data objects. ########################################
#    myPtr   = pydarn.sdio.radDataOpen(load_sDatetime,radar,eTime=load_fDatetime,channel=channel,cp=cp,fileType=fileType,filtered=boxCarFilter)
    myPtr   = load_fitacf(radar, load_sDatetime, eTime=load_fDatetime)
    dataObj = music.musicArray(myPtr,fovModel='GS')
    # myPtr.close()

    bad = False # Innocent until proven guilty.
    if hasattr(dataObj,'messages'):
        if 'No data for this time period.' in dataObj.messages:
            bad = True # At this point, proven guilty.
    
    if not bad:
        gl = None
        if np.size(gateLimits) == 2:
            if gateLimits[0] != None or gateLimits[1] !=None:
                if gateLimits[0] == None:
                    gl0 = min(dataObj.active.fov["gates"])
                else:
                    gl0 = gateLimits[0]
                if gateLimits[1] == None:
                    gl1 = max(dataObj.active.fov["gates"])
                else:
                    gl1 = gateLimits[1]
                gl = (gl0, gl1)

        if gl != None:
            defineLimits(dataObj,gateLimits=gl)

        bl = None
        if np.size(beamLimits) == 2:
            if beamLimits[0] != None or beamLimits[1] !=None:
                if beamLimits[0] == None:
                    bl0 = min(dataObj.active.fov["beams"])
                else:
                    bl0 = beamLimits[0]
                if beamLimits[1] == None:
                    bl1 = max(dataObj.active.fov["beams"])
                else:
                    bl1 = beamLimits[1]
                bl = (bl0, bl1)

        if bl != None:
            defineLimits(dataObj,beamLimits=bl)

        dataObj = checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)

    hdf5Path = get_hdf5_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
    saveMusicArrayToHDF5(dataObj, hdf5Path)

    return dataObj

def zeropad(dataObj):
    samp_per    = datetime.timedelta(seconds=dataObj.active.samplePeriod())
    time_delt   = np.max(dataObj.active.time) - np.min(dataObj.active.time)
    new_time = np.array(
    (dataObj.active.time - time_delt - samp_per).tolist() + \
    (dataObj.active.time).tolist() + \
    (dataObj.active.time + time_delt + samp_per).tolist()
    )

    size     = dataObj.active.data.shape[0]
    new_data = np.pad(dataObj.active.data,((size,size),(0,0),(0,0)),'constant')

    new_sig      = dataObj.active.copy('zeropad','Zero Padded Signal')
    new_sig.time = new_time
    new_sig.data = new_data

    new_sig.setMetadata(sTime=new_time.min())
    new_sig.setMetadata(eTime=new_time.max())

    new_sig.setActive()

def window_beam_gate(dataObj,dataSet='active',window='hann'):
    import scipy as sp

    currentData = getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    win = sp.signal.get_window(window,nrGates,fftbins=False)
    win.shape = (1,1,nrGates)

    new_sig      = dataObj.active.copy('windowed_gate','Windowed Gate Dimension')
    new_sig.data = win*dataObj.active.data
    new_sig.setActive()
    
    win = sp.signal.get_window(window,nrBeams,fftbins=False)
    win.shape = (1,nrBeams,1)

    new_sig      = dataObj.active.copy('windowed_beam','Windowed Beam Dimension')
    new_sig.data = win*dataObj.active.data
    new_sig.setActive()

def run_music(runfile_path,process_level='all'):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj = loadMusicArrayFromHDF5(musicObj_path)

    rad         = musicParams['radar']
    sDatetime   = musicParams['sDatetime']
    fDatetime   = musicParams['fDatetime']
    interpRes   = musicParams['interpRes']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    kx_max      = musicParams['kx_max']
    ky_max      = musicParams['ky_max']
    threshold   = musicParams['autodetect_threshold']
    neighborhood = musicParams['neighborhood']

    dataObj = checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)
    if not dataObj.active.metadata['good_period']:
        saveMusicArrayToHDF5(dataObj, musicObj_path)
        return

    dataObj.active.applyLimits()
    beamInterpolation(dataObj,dataSet='limitsApplied')

    determineRelativePosition(dataObj)

    timeInterpolation(dataObj,timeRes=interpRes)
    nan_to_num(dataObj)

    if not numtaps is None:
        filt = music.filter(dataObj, dataSet='active', numtaps=numtaps, cutoff_low=cutoff_low, cutoff_high=cutoff_high)

        outputDir   = musicParams['path']
        figsize    = (20,10)
        plotSerial = 999
        fig = plt.figure(figsize=figsize)
        filt.plotImpulseResponse(fig=fig)
        fileName = os.path.join(outputDir,'%03i_impulseResponse.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        plt.close()
    #    plotSerial = plotSerial + 1

        fig = plt.figure(figsize=figsize)
        filt.plotTransferFunction(fig=fig,xmax=0.004)
        fileName = os.path.join(outputDir,'%03i_transferFunction.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        plt.close()
    #    plotSerial = plotSerial + 1

    detrend(dataObj, dataSet='active')

    windowData(dataObj, dataSet='active')
#    window_beam_gate(dataObj)
    zeropad(dataObj)

    calculateFFT(dataObj)
    if process_level == 'fft':
        saveMusicArrayToHDF5(dataObj, musicObj_path)
        return
    calculateDlm(dataObj)
    calculateKarr(dataObj,kxMax=kx_max,kyMax=ky_max)
    detectSignals(dataObj,threshold=threshold,neighborhood=neighborhood)


    saveMusicArrayToHDF5(dataObj, musicObj_path)

def music_plot_all(runfile_path,process_level='all',rti_only=False):
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

    if rti_only:
        return

    if 'good_period' in dataObj.active.metadata:
        if not dataObj.active.metadata['good_period']:
            return

    fig = plt.figure(figsize=figsize)
    # ax  = fig.add_subplot(121)
    musicFan(dataObj,plotZeros=True,dataSet='originalFit',fig=fig,time=time,subplot_tuple=(1,2,1))
    # ax  = fig.add_subplot(122)
    musicFan(dataObj,plotZeros=True,dataSet='beamInterpolated',fig=fig,time=time,subplot_tuple=(1,2,2))
    fileName = os.path.join(outputDir,'%03i_beamInterp_fan.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    plotRelativeRanges(dataObj,time=time,fig=fig,dataSet='beamInterpolated')
    fileName = os.path.join(outputDir,'%03i_ranges.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    timeSeriesMultiPlot(dataObj,dataSet="DS002_beamInterpolated",dataSet2='DS001_limitsApplied',fig=fig)
    fileName = os.path.join(outputDir,'%03i_beamInterp.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    timeSeriesMultiPlot(dataObj,dataSet='timeInterpolated',dataSet2='beamInterpolated',fig=fig)
    fileName = os.path.join(outputDir,'%03i_timeInterp.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    timeSeriesMultiPlot(dataObj,fig=fig,dataSet="DS005_filtered")
    fileName = os.path.join(outputDir,'%03i_filtered.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    timeSeriesMultiPlot(dataObj,fig=fig)
    fileName = os.path.join(outputDir,'%03i_detrendedData.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    if musicParams.get('window_data'):
        fig = plt.figure(figsize=figsize)
        timeSeriesMultiPlot(dataObj,fig=fig)
        fileName = os.path.join(outputDir,'%03i_windowedData.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        plt.close()
        plotSerial = plotSerial + 1


    fig = plt.figure(figsize=figsize)
    spectrumMultiPlot(dataObj,fig=fig,xlim=(-0.0025,0.0025))
    fileName = os.path.join(outputDir,'%03i_spectrum.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    spectrumMultiPlot(dataObj,fig=fig,plotType='magnitude',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_magnitude.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    spectrumMultiPlot(dataObj,fig=fig,plotType='phase',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_phase.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    # ax  = fig.add_subplot(111)
    musicFan(dataObj,plotZeros=True,fig=fig,autoScale=True,time=time,subplot_tuple=(1,1,1))
    fileName = os.path.join(outputDir,'%03i_finalDataFan.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    musicRTP(dataObj,plotZeros=True,axis=ax,autoScale=True)
    fileName = os.path.join(outputDir,'%03i_finalDataRTI.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    fig = plt.figure(figsize=figsize)
    plotFullSpectrum(dataObj,fig=fig,xlim=(0,0.0015))
    fileName = os.path.join(outputDir,'%03i_fullSpectrum.png' % plotSerial)
    fig.savefig(fileName,bbox_inches='tight')
    plt.close()
    plotSerial = plotSerial + 1

    try:
        fig = plt.figure(figsize=figsize)
        plotDlm(dataObj,fig=fig)
        fileName = os.path.join(outputDir,'%03i_dlm_abs.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        plt.close()
        plotSerial = plotSerial + 1
    except:
        pass

    try:
        fig = plt.figure(figsize=(10,10))
        plotKarr(dataObj,fig=fig,maxSignals=25)
        fileName = os.path.join(outputDir,'%03i_karr.png' % plotSerial)
        fig.savefig(fileName,bbox_inches='tight')
        plt.close()
        plotSerial = plotSerial + 1
    except:
        pass

#    fig = plt.figure(figsize=(10,10))
#    pydarn.plotting.musicPlot.plotKarrDetected(dataObj,fig=fig)
#    fileName = os.path.join(outputDir,'%03i_karrDetected.png' % plotSerial)
#    fig.savefig(fileName,bbox_inches='tight')
#    plt.close()
#    plotSerial = plotSerial + 1

def music_plot_fan(runfile_path,time=None,fileName='fan.png',scale=None):
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

    figsize     = (20,10)
    plotSerial  = 0

    fig = plt.figure(figsize=figsize)
    # ax  = fig.add_subplot(121)
    musicFan(dataObj,plotZeros=True,dataSet='originalFit',fig=fig,time=time,scale=scale,subplot_tuple=(1,2,1))
    # ax  = fig.add_subplot(122)
    musicFan(dataObj,plotZeros=True,fig=fig,dataSet='beamInterpolated',time=time,scale=scale,subplot_tuple=(1,2,2))
    fullPath = os.path.join(outputDir,fileName)
    fig.savefig(fullPath,bbox_inches='tight')
    plt.close()

def music_plot_karr(runfile_path,fileName='karr.png',maxSignals=25):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj = loadMusicArrayFromHDF5(musicObj_path)

    fig = plt.figure(figsize=(10,10))
    plotKarr(dataObj,fig=fig,maxSignals=maxSignals)
    fig.savefig(fileName,bbox_inches='tight')
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

    # from musicRTI3 import musicRTI3

    fig     = plt.figure(figsize=figsize)
    axis    = fig.add_subplot(111)
    #    pydarn.plotting.musicPlot.musicRTI(dataObj
    musicRTP(dataObj
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
    plt.close(fig)

def get_default_rti_times(musicParams,dataObj=None,min_hours=8):
    #Set up suggested boundaries for RTI replotting.
    min_timedelta = datetime.timedelta(hours=min_hours)
    sTime   = musicParams.get('sDatetime',musicParams.get('sTime'))
    eTime   = musicParams.get('fDatetime',musicParams.get('eTime'))

    duration = eTime - sTime
    if duration < min_timedelta:
        center_time = sTime + duration/2
        min_time = center_time - min_timedelta/2
        max_time = center_time + min_timedelta/2
    else:
        min_time = sTime
        max_time = eTime

    return min_time,max_time


def get_default_gate_range(musicParams,dataObj=None,gate_buffer=10):
    min_gate = None
    max_gate = None

    if hasattr(dataObj,'messages'):
        if 'No data for this time period.' in dataObj.messages:
            dataObj = None

    if 'gateLimits' in musicParams:
        if musicParams['gateLimits'] is not None:
            if musicParams['gateLimits'][0] is not None:
                min_gate = musicParams['gateLimits'][0] - gate_buffer
                if dataObj is not None:
                    gts = dataObj.DS000_originalFit.fov["gates"]
                    if min_gate < min(gts): min_gate = min(gts)
            if musicParams['gateLimits'][1] is not None:
                max_gate = musicParams['gateLimits'][1] + gate_buffer
                if dataObj is not None:
                    gts = dataObj.DS000_originalFit.fov["gates"]
                    if max_gate > max(gts): max_gate = max(gts)

    return min_gate,max_gate

def get_default_beams(musicParams,dataObj=None,beams=[4,7,13]):
    if hasattr(dataObj,'messages'):
        if 'No data for this time period.' in dataObj.messages:
            dataObj = None

    if dataObj is not None:
        new_beam_list = []
        bms = dataObj.DS000_originalFit.fov["beams"]
        for beam in beams:
            if beam in bms:
                new_beam_list.append(beam)
            else:
                new_beam_list.append(bms[0])
    else:
        new_beam_list = beams
    return new_beam_list

def calculate_terminator(lats,lons,dates):
    lats    = np.array(lats)
    lons    = np.array(lons)
    dates   = np.array(dates)

    if lats.shape == (): lats.shape = (1,)
    if lons.shape == (): lons.shape = (1,)

    shape       = (len(dates),lats.shape[0],lats.shape[1])

    term_lats   = np.zeros(shape,dtype=float)
    term_tau    = np.zeros(shape,dtype=float)
    term_dec    = np.zeros(shape,dtype=float)

    terminator  = np.ones(shape,dtype=bool)

    for inx,date in enumerate(dates):
        term_tup = daynight_terminator(date, lons)
        term_lats[inx,:,:]  = term_tup[0]
        term_tau[inx,:,:]   = term_tup[1]
        term_dec[inx,:,:]   = term_tup[2]

    nh_summer = term_dec > 0
    nh_winter = term_dec < 0

    tmp         = lats[:]
    tmp.shape   = (1,tmp.shape[0],tmp.shape[1])
    lats_arr    = np.repeat(tmp,len(dates),axis=0)
    terminator[nh_summer] = lats_arr[nh_summer] < term_lats[nh_summer]
    terminator[nh_winter] = lats_arr[nh_winter] > term_lats[nh_winter]

    return terminator

def calculate_terminator_for_dataSet(dataObj,dataSet='active'):
    currentData = getDataSet(dataObj,dataSet)

    term_ctr    = calculate_terminator(currentData.fov["latCenter"],currentData.fov["lonCenter"],currentData.time)
    currentData.terminator = term_ctr

#    term_full    = calculate_terminator(currentData.fov.latFull,currentData.fov.lonFull,currentData.time)
#    currentData.fov.terminatorFull = term_full
    return dataObj

def plot_classification_variables(dataObj,dct,dataSet='active',filename='classification.png'):
    from matplotlib import dates as md
    currentData = getDataSet(dataObj,dataSet)

    fig = plt.figure(figsize=(10,8))

    ax  = fig.add_subplot(2,1,1)
    y_vals = np.sum(currentData.terminator,axis=(1,2)) / float( len(currentData.fov["latCenter"]) )
    ax.plot(currentData.time,y_vals,'o-')

    fontdict = {'size':14,'weight':'bold'}
    ax.set_title('{0}'.format(os.path.basename(filename)))
    ax.set_xlabel('Time [UT]',fontdict=fontdict)
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

    ax.set_ylabel('Terminator Cells',fontdict=fontdict)
    ax.set_ylim(0,1.1)
    
#    pct_term = 100.*np.sum(currentData.terminator)/float(currentData.terminator.size)
    text = []
    text.append('Term Frac: {0:0.3f}'.format(dct['terminator_fraction']))
    text.append('RTI Frac : {0:0.3f}'.format(dct['orig_rti_fraction']))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    box_fontdict = {'size':20,'weight':'bold','family':'monospace'}
    ax.text(0.95,0.95,'\n'.join(text),transform=ax.transAxes,fontdict=box_fontdict,va='top',ha='right',bbox=props)

    ax  = fig.add_subplot(2,1,2)

    if hasattr(currentData,'spectrum'):
        y_vals      = np.sum(np.abs(currentData.spectrum)**2,axis=(1,2))
        ax.plot(currentData.freqVec*1000.,y_vals,'-o')

        # Get top frequencies.
        nr_top = dct['nr_dom_freqs_used']
        pos_bools = currentData.freqVec >= 0
        
        pos_psd = [(freq,val) for freq,val in zip(currentData.freqVec[pos_bools],y_vals[pos_bools])]
        pos_psd_srt = sorted(pos_psd,key=lambda x: x[1])[::-1]

        x_vals, y_vals = list(zip(*pos_psd_srt[:nr_top]))
        ax.plot(np.array(x_vals)*1000.,y_vals,'o',color='red')

    ax.set_xlim(xmin=0)

    ax.set_xlabel('Frequency [mHz]',fontdict=fontdict)
    ax.set_ylabel('Power SD',fontdict=fontdict)

    text = []
    text.append('Norm Spec: {0:0.3f}'.format(dct['norm_total_spec']))
    text.append('Dom Spec Ratio: {0:0.3f}'.format(dct['dom_spectral_ratio']))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.95,0.95,'\n'.join(text),transform=ax.transAxes,fontdict=box_fontdict,va='top',ha='right',bbox=props)

    fig.tight_layout()
    fig.savefig(filename,bbox_inches='tight')
    print(filename)
    plt.close(fig)
