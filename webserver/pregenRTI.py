#!/usr/bin/env python
import datetime
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import numpy as np

import pydarn

radar = 'bks'
param = 'power'
sTime = datetime.datetime(2011,1,1)
eTime = datetime.datetime(2011,1,2)

currentTime = sTime
while currentTime < eTime:
    gwDay = currentTime.strftime('%Y%m%d')

    #Build the path of the RTI plot and call the plotting routine.
    d = 'static/rti'
    outputFile = d+'/'+gwDay+'.'+radar+'.'+param+'rti.png'
    try:
        os.makedirs(d)
    except:
        pass

#    if not os.path.exists(outputFile):
    if True:
        try:
            font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 22}
            matplotlib.rc('font', **font)
            tick_size   = 16

            xticks  = []
            hours   = np.arange(0,26,2) #Put a tick every 2 hours.
            for hour in hours:
                tmp = currentTime + datetime.timedelta(hours=float(hour))
                xticks.append(tmp)
            axvlines  = xticks[:]
            fig = Figure(figsize=(20,10))
            fig = pydarn.plotting.rti.plotRti(currentTime,radar,params=[param],show=False,retfig=True,
                    figure=fig,xtick_size=tick_size,ytick_size=tick_size,xticks=xticks,axvlines=axvlines,plotTerminator=True)
            canvas = FigureCanvasAgg(fig)
            fig.savefig(outputFile)
            fig.clf()
        except:
            pass
    #    rti.plotRti(currentTime,radar,params=[param],outputFile=outputFile)
    currentTime = currentTime + datetime.timedelta(hours=24)
