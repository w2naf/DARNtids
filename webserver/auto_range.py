#!/usr/bin/env python
############################################
# This code adds davitpy to your python path
# Eventually, this won't be necessary
# import sys
# sys.path.append('/davitpy')
############################################

import os,shutil
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from scipy import stats as stats

#pickle.dump(dataObj,open('dataObj.p','wb'))

def auto_range(dataObj,runParams,figsize = (20,7),outputDir = 'output',plot=False):

    sTime   = runParams['sDatetime']
    eTime   = runParams['fDatetime']
    radar   = runParams['radar']

#    fileName    = '/data/mstid/statistics/data/music_data/main_run/bks/20101119.1400-20101119.1600/bks-20101119.1400-20101119.1600.p'
#    dataObj     = pickle.load(open(fileName,'rb'))
    # Auto-ranging code ############################################################
    currentData = dataObj.DS000_originalFit
    timeInx = np.where(np.logical_and(currentData.time >= sTime,currentData.time <= eTime))[0]

    bins    = currentData.fov.gates
    dist    = np.nansum(np.nansum(currentData.data[timeInx,:,:],axis=0),axis=0)
    dist    = np.nan_to_num(dist / np.nanmax(dist))

    nrPts   = 1000
    distArr = np.array([],dtype=np.int)
    for rg in range(len(bins)):
        gate    = bins[rg]
        nrGate  = np.floor(dist[rg]*nrPts)
        distArr = np.concatenate([distArr,np.ones(nrGate,dtype=np.int)*gate])

#    #sudo pip install scikit-learn
#    import sklearn.mixture
#    gmm     = sklearn.mixture.GMM(n_components=10,covariance_type='spherical')
#    #gmm     = sklearn.mixture.GMM(n_components=10,covariance_type='spherical')
#    #gmm     = sklearn.mixture.DPGMM(n_components=10,covariance_type='spherical')
#    fitted  = gmm.fit(np.nan_to_num(distArr))

    hist,bins           = np.histogram(distArr,bins=bins,density=True)
    hist                = sp.signal.medfilt(hist,kernel_size=11)

    arg_max = np.argmax(hist)
    max_val = hist[arg_max]
    thresh  = 0.18

    good    = [arg_max]
    #Search connected lower
    search_inx  = np.where(bins[:-1] < arg_max)[0]
    search_inx.sort()
    search_inx  = search_inx[::-1]
    for inx in search_inx:
        if hist[inx] > thresh*max_val:
            good.append(inx)
        else:
            break

    #Search connected upper
    search_inx  = np.where(bins[:-1] > arg_max)[0]
    search_inx.sort()
    for inx in search_inx:
        if hist[inx] > thresh*max_val:
            good.append(inx)
        else:
            break

    good.sort() 

    min_range   = min(good)
    max_range   = max(good)

    #Check for and correct bad start gate (due to GS mapping algorithm)
    bad_range   = np.max(np.where(dataObj.DS000_originalFit.fov.slantRCenter < 0)[1])
    if min_range <= bad_range: min_range = bad_range+1

    dataObj.DS000_originalFit.metadata['gateLimits'] = (min_range,max_range)

    if plot:
        # Make some plots. #############################################################
        if not os.path.exists(outputDir): os.makedirs(outputDir)

        file_name   = '.'.join([radar,sTime.strftime('%Y%m%d.%H%M'),eTime.strftime('%Y%m%d.%H%M'),'rangeDist','png'])

        font = {'weight':'normal','size':12}
        matplotlib.rc('font',**font)
        fig     = plt.figure(figsize=figsize)
    #    axis    = fig.add_subplot(121)
        axis    = fig.add_subplot(221)

        axis.bar(bins[:-1],hist)
        axis.bar(bins[good],hist[good],color='r')

    #    hist,bins,patches   = axis.hist(distArr,bins=bins,normed=1)
    #    for xx in xrange(fitted.n_components):
    #        mu      = fitted.means_[xx]
    #        sigma   = np.sqrt(fitted.covars_[xx])
    #        y       = stats.norm.pdf(bins,mu,sigma)
    #        axis.plot(bins,y)

        axis.set_xlabel('Range Gate')
        axis.set_ylabel('Normalized Weight')
        axis.set_title(file_name)

        axis    = fig.add_subplot(223)
        axis.plot(bins[:-1],np.cumsum(hist))
        axis.set_xlabel('Range Gate')
        axis.set_ylabel('Power CDF')

        axis    = fig.add_subplot(122)
        from musicRTI3 import musicRTI3
        musicRTI3(dataObj
            , dataSet='originalFit'
    #        , beams=beam
            , xlim=None
            , ylim=None
            , coords='gate'
            , axis=axis
            , plotZeros=True
            , xBoundaryLimits=(sTime,eTime)
    #        , axvlines = axvlines
    #        , autoScale=autoScale
            )
       ################################################################################ 
        fig.tight_layout(w_pad=5.0)
        fig.savefig(os.path.join(outputDir,file_name))

    return (min_range,max_range)
