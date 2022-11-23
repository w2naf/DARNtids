import os
import sys
import utils
import shutil
import tempfile
import pickle
import datetime
import glob

import numpy as np
from scipy.io.idl import readsav
from scipy import signal

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import pydarn
import pydarn.sdio
import pydarn.proc.music as music

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

def get_output_path(radar,sDatetime,fDatetime,create=True):
    lst = []
    lst.append('/data')
    lst.append('pymusic')
    lst.append(radar.lower())
    lst.append('-'.join([sDatetime.strftime('%Y%m%d.%H%M'),fDatetime.strftime('%Y%m%d.%H%M')]))
    path = os.path.join(*lst)
    if create:
        try:
            os.makedirs(path)
        except:
            pass
    return path

def get_pickle_name(radar,sDatetime,fDatetime,getPath=False,createPath=False):
    fName = ('-'.join([radar.lower(),sDatetime.strftime('%Y%m%d.%H%M'),fDatetime.strftime('%Y%m%d.%H%M')]))+'.p'
    if getPath:
        path = get_output_path(radar,sDatetime,fDatetime,create=createPath)
        fName = os.path.join(path,fName)
    return fName

class Runfile(object):
    def __init__(self,radar,sDatetime,fDatetime, runParamsDict):
        picklePath = get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
        picklePath = picklePath[:-1] + 'runfile.p'
        
        self.runParams = {}
        for key,value in runParamsDict.items():
            self.runParams[key] = value

        self.runParams['runfile_path'] = picklePath
        
        pickle.dump(self,open(picklePath,'wb'))
        

def load_runfile_path(path):
        try:
            runFile = pickle.load(open(path,'rb'))
        except:
            runFile = None
        
        return runFile


def load_runfile(radar,sDatetime,fDatetime):
        picklePath = get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
        picklePath = picklePath[:-1] + 'runfile.p'

        return load_runfile_path(picklePath)
        
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
    myPtr   = pydarn.sdio.radDataOpen(load_sDatetime,radar,eTime=load_fDatetime)
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

    picklePath = get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=True)
    pickle.dump(dataObj,open(picklePath,'wb'))

    return dataObj


def plot_music_rti(dataObj
        , dataSet='active'
        , beam=7
        , xlim=None
        , ylim=None
        , coords='gate'
        , fileName='rti.png'
        , scale=None
        , plotZeros=False
        , xBoundaryLimits=None
        , yBoundaryLimits=None
        , autoScale=False
        , figsize = (20,10)
        ):

    fig = Figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicRTI(dataObj
        , dataSet=dataSet
        , beam=beam
        , xlim=xlim
        , ylim=ylim
        , coords=coords
        , axis=ax
        , scale=scale
        , plotZeros=plotZeros
        , xBoundaryLimits=xBoundaryLimits
        , yBoundaryLimits=yBoundaryLimits
        , autoScale=autoScale
        )
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')


def run_music(runfile_path):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

    rad         = musicParams['radar']
    interpRes   = musicParams['interpRes']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    kx_max      = musicParams['kx_max']
    ky_max      = musicParams['ky_max']
    threshold   = musicParams['autodetect_threshold']
    neighborhood = musicParams['neighborhood']

    dataObj.active.applyLimits()
    music.beamInterpolation(dataObj,dataSet='limitsApplied')

    music.determineRelativePosition(dataObj)

    music.timeInterpolation(dataObj,timeRes=interpRes)
    music.nan_to_num(dataObj)

    filt = music.filter(dataObj, dataSet='active', numtaps=numtaps, cutoff_low=cutoff_low, cutoff_high=cutoff_high)

    outputDir   = musicParams['path']
    figsize    = (20,10)
    plotSerial = 999
    fig = Figure(figsize=figsize)
    filt.plotImpulseResponse(fig=fig)
    fileName = os.path.join(outputDir,'%03i_impulseResponse.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
#    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    filt.plotTransferFunction(fig=fig,xmax=0.004)
    fileName = os.path.join(outputDir,'%03i_transferFunction.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
#    plotSerial = plotSerial + 1

    music.detrend(dataObj, dataSet='active')

    if musicParams['window_data']:
        music.windowData(dataObj, dataSet='active')

    music.calculateFFT(dataObj)

    music.calculateDlm(dataObj)

    music.calculateKarr(dataObj,kxMax=kx_max,kyMax=ky_max)

    music.detectSignals(dataObj,threshold=threshold,neighborhood=neighborhood)

    pickle.dump(dataObj,open(musicObj_path,'wb'))

def music_plot_all(runfile_path):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

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

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicRTI(dataObj,plotZeros=True,axis=ax,dataSet='originalFit',
            xBoundaryLimits=(sDatetime,fDatetime))
    fileName = os.path.join(outputDir,'%03i_originalFit_RTI.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

#    fig = plt.figure(figsize=figsize)
#    ax  = fig.add_subplot(121)
#    pydarn.plotting.musicPlot.musicFan(dataObj   ,plotZeros=True,dataSet='originalFit',axis=ax)
#    ax  = fig.add_subplot(122)
#    pydarn.plotting.musicPlot.musicFan(dataObj_IS,plotZeros=True,dataSet='originalFit',axis=ax)
#    fig.savefig(outputDir+'/%03i_range_comparison.png' % plotSerial)
#    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    ax  = fig.add_subplot(121)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,dataSet='originalFit',axis=ax,time=time)
    ax  = fig.add_subplot(122)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,dataSet='beamInterpolated',time=time)
    fileName = os.path.join(outputDir,'%03i_beamInterp_fan.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.plotRelativeRanges(dataObj,time=time,fig=fig,dataSet='beamInterpolated')
    fileName = os.path.join(outputDir,'%03i_ranges.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,dataSet="DS002_beamInterpolated",dataSet2='DS001_limitsApplied',fig=fig)
    fileName = os.path.join(outputDir,'%03i_beamInterp.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,dataSet='timeInterpolated',dataSet2='beamInterpolated',fig=fig)
    fileName = os.path.join(outputDir,'%03i_timeInterp.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig,dataSet="DS005_filtered",dataSet2="timeInterpolated")
    fileName = os.path.join(outputDir,'%03i_filtered.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig,dataSet2='filtered')
    fileName = os.path.join(outputDir,'%03i_detrendedData.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    if musicParams['window_data']:
        fig = Figure(figsize=figsize)
        pydarn.plotting.musicPlot.timeSeriesMultiPlot(dataObj,fig=fig)
        fileName = os.path.join(outputDir,'%03i_windowedData.png' % plotSerial)
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
        plotSerial = plotSerial + 1


    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,xlim=(-0.0025,0.0025))
    fileName = os.path.join(outputDir,'%03i_spectrum.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,plotType='magnitude',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_magnitude.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.spectrumMultiPlot(dataObj,fig=fig,plotType='phase',xlim=(0,0.0025))
    fileName = os.path.join(outputDir,'%03i_phase.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,autoScale=True,time=time)
    fileName = os.path.join(outputDir,'%03i_finalDataFan.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicRTI(dataObj,plotZeros=True,axis=ax,autoScale=True)
    fileName = os.path.join(outputDir,'%03i_finalDataRTI.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.plotFullSpectrum(dataObj,fig=fig,xlim=(0,0.0015))
    fileName = os.path.join(outputDir,'%03i_fullSpectrum.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

    fig = Figure(figsize=figsize)
    pydarn.plotting.musicPlot.plotDlm(dataObj,fig=fig)
    fileName = os.path.join(outputDir,'%03i_dlm_abs.png' % plotSerial)
    plotSerial = plotSerial + 1

    fig = Figure(figsize=(10,10))
    pydarn.plotting.musicPlot.plotKarr(dataObj,fig=fig,maxSignals=25)
    fileName = os.path.join(outputDir,'%03i_karr.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

#    fig = Figure(figsize=(10,10))
#    pydarn.plotting.musicPlot.plotKarrDetected(dataObj,fig=fig)
#    fileName = os.path.join(outputDir,'%03i_karrDetected.png' % plotSerial)
#    canvas = FigureCanvasAgg(fig)
#    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
#    plotSerial = plotSerial + 1

def music_plot_rti(runfile_path):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

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

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)
    pydarn.plotting.musicPlot.musicRTI(dataObj,plotZeros=True,axis=ax,dataSet='originalFit',
            xBoundaryLimits=(sDatetime,fDatetime))
    fileName = os.path.join(outputDir,'%03i_originalFit_RTI.png' % plotSerial)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
    plotSerial = plotSerial + 1

def music_plot_fan(runfile_path,time=None,fileName='fan.png',scale=None):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

    rad         = musicParams['radar']
    interpRes   = musicParams['interpRes']
    numtaps     = musicParams['filter_numtaps']
    cutoff_low  = musicParams['filter_cutoff_low']
    cutoff_high = musicParams['filter_cutoff_high']
    outputDir   = musicParams['path']

    figsize     = (20,10)
    plotSerial  = 0

    fig = Figure(figsize=figsize)
    ax  = fig.add_subplot(121)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,dataSet='originalFit',axis=ax,time=time,scale=scale)
    ax  = fig.add_subplot(122)
    pydarn.plotting.musicPlot.musicFan(dataObj,plotZeros=True,axis=ax,dataSet='beamInterpolated',time=time,scale=scale)
    fullPath = os.path.join(outputDir,fileName)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fullPath,format='png',facecolor='white',edgecolor='white')

def music_plot_karr(runfile_path,fileName='karr.png',maxSignals=25):
    runFile         = load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

    fig = Figure(figsize=(10,10))
    pydarn.plotting.musicPlot.plotKarr(dataObj,fig=fig,maxSignals=maxSignals)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fileName,format='png',facecolor='white',edgecolor='white')
