#!/usr/bin/env python
#From http://flask.pocoo.org/docs/tutorial/setup/#tutorial-setup
#all the imports
import pymongo
from bson.objectid import ObjectId

import datetime
import os
import sys

import numpy as np
from scipy.io.idl import readsav
from scipy import signal

from davitpy import utils

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

mongo         = pymongo.MongoClient()
db            = mongo.mstid

clr_0   = 'blue'
lbl_0   = 'MSTID Not Obs.'
alpha_0 = 0.6

clr_1   = 'red'
lbl_1   = 'MSTID Observed'
alpha_1 = 0.6

def kpBins(width=0.1):
  kpCtr = [0.,0.33,0.67,1.,1.33,1.67,2.,2.33,2.67,3.,3.33,3.67,4.,4.33,4.67,5.,5.33,5.67,6.,6.33,6.67,7.,7.33,7.67,8.,8.33,8.67,9]

  bins = []
  for kp in kpCtr:
    bins.append(kp-width)
    bins.append(kp+width)
  return bins
    
def get_KP(timeList):
  '''Download KP data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the KP dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  kpDict = {}
  for time in timeList:
    kp_man_rec= db['kpManual'].find_one({'date':time})
    if kp_man_rec == None:
      #Find the Kp record immediately before the time of interest.
      #Also, make sure that the returned value is within 3.5 hours of
      #the time of interest.
      cursor = db['kp'].find({'date': {'$lte': time}}).sort('date',-1)
      record = None

      if  cursor.count() != 0:
        record = ([x for x in cursor])
        record = record[0]
      else:
        record = None

      if record != None:
        if ( (time-record['date']) >= datetime.timedelta(hours=3)):
          record = None
        else: 
          ap = record['ap']
          kp = record['kp']

      #If nothing satisfies the search, go fetch Kp and populate the local database.
      if record == None:
        year  = time.year
        month = time.month
        dy    = time.day
        day   = datetime.datetime(year,month,dy)
        kpObj = gme.ind.kp.readKp(sTime=day,eTime=day)

        if kpObj != None:
          for inx in range(8):
            db['kp'].insert({'date':day, 'kp': kpObj[0].kp[inx], 'ap':kpObj[0].ap[inx]})
            day = day + datetime.timedelta(hours=3)

        #Now, go back and try to find the value of interest again.
        cursor = db['kp'].find({'date': {'$lte': time}}).sort('date',-1)
        record = None

        if  cursor.count() != 0:
          record = ([x for x in cursor])
          record = record[0]
        else:
          record = None
        if record != None:
          condition = ( (time-record['date']) >= datetime.timedelta(hours=3))
          if condition:
            record = None
          else:
            ap = record['ap']
            kp = record['kp']
        #If things still fail, just return the data as np.nan.
        if record == None:
            ap = 'NaN'
            kp = 'NaN'
      kp_man_rec = {'date':time,'ap':ap,'kp':kp}
      db['kpManual'].insert(kp_man_rec)

    if kp_man_rec['ap'] == 'NaN':
      ap = np.nan
    else:
      ap = kp_man_rec['ap']
    kp = kp_man_rec['kp']

    kpDict[time] = {'ap':ap, 'kp':kp}
  return kpDict

def kpHistogram(gwDays,secondPop=None,title='Kp Distribution',kind='mean',outFName='static/output/kp.png'):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig     = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx = 0
  ys = []
  xs = []
  for lst in dataLists:
    if lst==[]: continue
    sTime = min(lst)
    eTime = max(lst)

#    kpDict = get_KP_day(lst)
    kpDict = get_KP(lst)

    ap      = [kpDict[day]['ap'] for day in lst]
    kp      = apToKp(ap)
#    kp_str  = [kpDict[day]['kp'] for day in lst]
#    kp_str_float = {'0': 0., '0+': 0.33, '1-': 0.67, '1' : 1.00, '1+': 1.33, '2-': 1.67, '2' : 2.00, '2+': 2.33, '3-': 2.67, '3' : 3.00, '3+': 3.33, '4-': 3.67, '4' : 4.00, '4+': 4.33, '5-': 5.67, '5' : 5.00, '5+': 5.33, '6-': 5.67, '6' : 6.00, '6+': 6.33, '7-': 6.67, '7' : 7.00, '7+': 7.33, '8-': 7.67, '8' : 8.00, '8+': 8.33, '9-': 8.67, '9' : 9.00}
#    kp = [kpDict[day]['kp_'+kind] for day in lst]

    n, bins = np.histogram(kp, kpBins(0.15),density=True)
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    xs.append(left)
    xs.append(right)
    ys.append(top)

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    if inx == 0 and len(dataLists) == 2:
      clr   = clr_0
      lbl   = lbl_0
      alpha = alpha_0
    else:
      clr   = clr_1
      lbl   = lbl_1
      alpha = alpha_1
    patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
    ax.add_patch(patch)
    inx = inx+1

  # update the view limits
#  ax.set_xlim(left[0], right[-1])
  ax.set_xlim(-0.20, 9.2)
  if ys != []:
    ax.set_ylim(0,np.max(np.array(ys)))
    ax.set_xlabel('Kp Index')
#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_xticks(range(10))
  ax.set_title(title)
  ax.legend()

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_AE(dayList):
  '''Download AE data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the AE dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  aeDict = {}
  for day in dayList:
    record = db['aeManual'].find_one({'date':day})

    if record == None:
      sTime = day - datetime.timedelta(hours=2)
      eTime = day
      aeList = gme.ind.ae.readAe(sTime=sTime, eTime=eTime,res=1)
      oneDayList = []
      for ae in aeList:
        if ae.time >= sTime and ae.time < eTime: oneDayList.append(ae.ae)
      if oneDayList != []:
        record = {'date':day,'mean':np.mean(oneDayList),'max':np.max(oneDayList)}
      else:
        record = {'date':day,'mean':'NaN','max':'NaN'}
      db['aeManual'].insert(record)
    if record['mean']=='NaN':
      aeDict[day] = {'mean':np.nan, 'max':np.nan}
    else:
      aeDict[day] = {'mean':record['mean'], 'max':record['max']}
  return aeDict
    
def aeHistogram(gwDays,secondPop=None,title='AE Distribution',kind='mean',outFName='static/output/ae.png'):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig     = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  aeLst   = []
  maxLst  = []
  for lst in dataLists:
    if lst==[]: continue
    sTime = min(lst)
    eTime = max(lst)

#    aeDict = get_AE_day(lst)
    aeDict = get_AE(lst)

    tmpLst = [aeDict[day][kind] for day in lst]
    aeLst.append(tmpLst)
    maxLst.append(max(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    if mx <= 250:
      mxr = 250
    elif mx <= 750:
      mxr = 500
    elif mx <= 500:
      mxr = 750
    elif mx <= 1000:
      mxr = 1000
    elif mx <= 1250:
      mxr = 1250
    elif mx <= 1500:
      mxr = 1500
    elif mx <= 1750:
      mxr = 1750
    elif mx <= 2000:
      mxr = 2000
    elif mx <= 2250:
      mxr = 2250
    elif mx <= 2500:
      mxr = 2500
    else:
      mxr = mx

    rnge = [0,mxr]
    for ae in aeLst:
      n, bins = np.histogram(ae,bins=25,range=rnge,density=True)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

  if kind=='mean':
    ax.set_xlabel('Two Hour Average AE Index')
  if kind=='max':
    ax.set_xlabel('Two Hour Maximum AE Index')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_title(title)
  ax.legend()

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_SYMH(dayList):
  '''Download SYMH data and calculate an average and absolute maximum for each two hour period in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the SYMH dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  symhDict = {}

  for day in dayList:
    record = db['symhManual'].find_one({'date':day})

    if record == None:
      sTime = day - datetime.timedelta(hours=2)
      eTime = day
      symhList = gme.ind.symasy.readSymAsy(sTime=day, eTime=eTime)
      oneDayList = []
      for symh in symhList:
        if symh.time >= sTime and symh.time <= eTime: oneDayList.append(symh.symh)
      if oneDayList != []:
        #Find the absolute value maximum, but return the real value.
        mx = np.max(np.abs(oneDayList))
        idx = (np.where(np.abs(oneDayList) == mx))[0]
        if len(idx) == 1:
          mx = oneDayList[idx]
        else:
          mx = oneDayList[idx[0]]
        record = {'date':day,'mean':np.mean(oneDayList),'max':mx}
      else:
        record = {'date':day,'mean':'NaN','max':'NaN'}
      db['symhManual'].insert(record)
    if record['mean']=='NaN':
      symhDict[day] = {'mean':np.nan, 'max':np.nan}
    else:
      symhDict[day] = {'mean':record['mean'], 'max':record['max']}
  return symhDict
    
def symhHistogram(gwDays,secondPop=None,title='SYM-H Distribution',kind='mean',outFName='static/output/symh.png'):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  symhLst = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    symhDict = get_SYMH(lst)

    tmpLst = [symhDict[day][kind] for day in lst]
    symhLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)

    rnge = [mn,mx]

    for symh in symhLst:
      n, bins = np.histogram(symh,bins=25,range=rnge,density=True)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

#  if kind=='mean':
#    ax.set_xlabel('Daily Average SYM-H Index')
#  if kind=='max':
  ax.set_xlabel('SYM-H Index')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_title(title)
  ax.legend(loc='upper left')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def apDayList(kpList):
  apDict  = {}
  for kp in kpList:
    apDict[kp.time] = kp.apMean
  return apDict

def get_ground_scatter(dayList,radar):
  ''' 
  dayList is a list of datetime.datetime objects.
  '''
  gsDict = {}
  for day in dayList:
    record = db['ground_scatter_manual'].find_one({'date':day,'radar':radar})

    if record == None:
      print day
      sTime = day 
      eTime = day + datetime.timedelta(hours=2)

      cursor      = db['ground_scatter'].find({'radar':radar, 'date': {'$gte': sTime, '$lt':eTime}})
      gsObj_list  = [x for x in cursor]

      oneDayList = []
      for gsObj in gsObj_list:
        oneDayList.append(gsObj['gscat'])

      if oneDayList != []:
        gscat = float(np.sum(oneDayList))
      else:
        gscat = 'NaN'
      record= {'date':day, 'radar':radar, 'gscat': gscat}
      db['ground_scatter_manual'].insert(record)

    if record['gscat']=='NaN':
      gsDict[day] = {'gscat': np.nan}
    else:
      gsDict[day] = {'gscat': record['gscat']}
  return gsDict

def get_AE(dayList):
  '''Download AE data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the AE dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  aeDict = {}
  for day in dayList:
    record = db['aeManual'].find_one({'date':day})

    if record == None:
      sTime = day - datetime.timedelta(hours=2)
      eTime = day
      aeList = gme.ind.ae.readAe(sTime=sTime, eTime=eTime,res=1)
      oneDayList = []
      for ae in aeList:
        if ae.time >= sTime and ae.time < eTime: oneDayList.append(ae.ae)
      if oneDayList != []:
        record = {'date':day,'mean':np.mean(oneDayList),'max':np.max(oneDayList)}
      else:
        record = {'date':day,'mean':'NaN','max':'NaN'}
      db['aeManual'].insert(record)
    if record['mean']=='NaN':
      aeDict[day] = {'mean':np.nan, 'max':np.nan}
    else:
      aeDict[day] = {'mean':record['mean'], 'max':record['max']}
  return aeDict
    
def aeHistogram(gwDays,secondPop=None,title='AE Distribution',kind='mean',outFName='static/output/ae.png'):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig     = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  aeLst   = []
  maxLst  = []
  for lst in dataLists:
    if lst==[]: continue
    sTime = min(lst)
    eTime = max(lst)

#    aeDict = get_AE_day(lst)
    aeDict = get_AE(lst)

    tmpLst = [aeDict[day][kind] for day in lst]
    aeLst.append(tmpLst)
    maxLst.append(max(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    if mx <= 250:
      mxr = 250
    elif mx <= 750:
      mxr = 500
    elif mx <= 500:
      mxr = 750
    elif mx <= 1000:
      mxr = 1000
    elif mx <= 1250:
      mxr = 1250
    elif mx <= 1500:
      mxr = 1500
    elif mx <= 1750:
      mxr = 1750
    elif mx <= 2000:
      mxr = 2000
    elif mx <= 2250:
      mxr = 2250
    elif mx <= 2500:
      mxr = 2500
    else:
      mxr = mx

    rnge = [0,mxr]
    for ae in aeLst:
      n, bins = np.histogram(ae,bins=25,range=rnge,density=True)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

  if kind=='mean':
    ax.set_xlabel('Two Hour Average AE Index')
  if kind=='max':
    ax.set_xlabel('Two Hour Maximum AE Index')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_title(title)
  ax.legend()

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_SYMH(dayList):
  '''Download SYMH data and calculate an average and absolute maximum for each two hour period in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the SYMH dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  symhDict = {}

  for day in dayList:
    record = db['symhManual'].find_one({'date':day})

    if record == None:
      sTime = day - datetime.timedelta(hours=2)
      eTime = day
      symhList = gme.ind.symasy.readSymAsy(sTime=day, eTime=eTime)
      oneDayList = []
      for symh in symhList:
        if symh.time >= sTime and symh.time <= eTime: oneDayList.append(symh.symh)
      if oneDayList != []:
        #Find the absolute value maximum, but return the real value.
        mx = np.max(np.abs(oneDayList))
        idx = (np.where(np.abs(oneDayList) == mx))[0]
        if len(idx) == 1:
          mx = oneDayList[idx]
        else:
          mx = oneDayList[idx[0]]
        record = {'date':day,'mean':np.mean(oneDayList),'max':mx}
      else:
        record = {'date':day,'mean':'NaN','max':'NaN'}
      db['symhManual'].insert(record)
    if record['mean']=='NaN':
      symhDict[day] = {'mean':np.nan, 'max':np.nan}
    else:
      symhDict[day] = {'mean':record['mean'], 'max':record['max']}
  return symhDict
    
def symhHistogram(gwDays,secondPop=None,title='SYM-H Distribution',kind='mean',outFName='static/output/symh.png'):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  symhLst = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    symhDict = get_SYMH(lst)

    tmpLst = [symhDict[day][kind] for day in lst]
    symhLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)

    rnge = [mn,mx]

    for symh in symhLst:
      n, bins = np.histogram(symh,bins=25,range=rnge,density=True)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

#  if kind=='mean':
#    ax.set_xlabel('Daily Average SYM-H Index')
#  if kind=='max':
  ax.set_xlabel('SYM-H Index')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_title(title)
  ax.legend(loc='upper left')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def apToKp(apList):
  #From table at en.wikipedia.org/wiki/K-Index
  if np.size(apList) == 1: apList = [apList]
  kpList = []
  for ap in apList:
    if ap < 2:
      kp = 0.
    elif ap < 3:
      kp = 0.33
    elif ap < 4:
      kp = 0.67
    elif ap < 5:
      kp = 1.
    elif ap < 6:
      kp = 1.33
    elif ap < 7:
      kp = 1.67
    elif ap < 9:
      kp = 2.
    elif ap < 12:
      kp = 2.33
    elif ap < 15:
      kp = 2.67
    elif ap < 18:
      kp = 3.
    elif ap < 22:
      kp = 3.33
    elif ap < 27:
      kp = 3.67
    elif ap < 32:
      kp = 4.
    elif ap < 39:
      kp = 4.33
    elif ap < 48:
      kp = 4.67
    elif ap < 56:
      kp = 5.
    elif ap < 67:
      kp = 5.33
    elif ap < 80:
      kp = 5.67
    elif ap < 94:
      kp = 6.
    elif ap < 111:
      kp = 6.33
    elif ap < 132:
      kp = 6.67
    elif ap < 154:
      kp = 7.
    elif ap < 179:
      kp = 7.33
    elif ap < 207:
      kp = 7.67
    elif ap < 236:
      kp = 8.
    elif ap < 300:
      kp = 8.33
    elif ap < 400:
      kp = 8.67
    else:
      kp = 9
    kpList.append(kp)
  if np.size(kpList) == 1: kpList = kpList[0]
  return kpList

def apDayList(kpList):
  apDict  = {}
  for kp in kpList:
    apDict[kp.time] = kp.apMean
  return apDict

def get_ground_scatter(dayList,radar,string_nan=False):
  ''' 
  dayList is a list of datetime.datetime objects.
  '''
  gsDict = {}
  for day in dayList:
    record = db['ground_scatter_manual'].find_one({'date':day,'radar':radar})

    if record == None:
      print day
      sTime = day 
      eTime = day + datetime.timedelta(hours=2)

      cursor      = db['ground_scatter'].find({'radar':radar, 'date': {'$gte': sTime, '$lt':eTime}})
      gsObj_list  = [x for x in cursor]

      oneDayList = []
      for gsObj in gsObj_list:
        oneDayList.append(gsObj['gscat'])

      if oneDayList != []:
        gscat = float(np.sum(oneDayList))
      else:
        gscat = 'NaN'
      record= {'date':day, 'radar':radar, 'gscat': gscat}
      db['ground_scatter_manual'].insert(record)

    if record['gscat']=='NaN' and string_nan == False:
      gsDict[day] = {'gscat': np.nan}
    else:
      gsDict[day] = {'gscat': record['gscat']}
  return gsDict

def ground_scatter_histogram(gwDays,radar,secondPop=None,title='Ground Scatter Distribution',kind='mean',outFName='static/output/ground_scatter.png',max_range=None):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  gsLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    gsDict = get_ground_scatter(lst,radar)

    tmpLst = [gsDict[day]['gscat'] for day in lst]
    gsLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)

    if max_range != None: mx = max_range
    rnge = [mn,mx]

    for gs in gsLst:
      n, bins = np.histogram(gs,bins=25,range=rnge,density=False)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
#    ax.set_xlim(left[0], right[-1])
    ax.set_xlim(0, right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

#  if kind=='mean':
#    ax.set_xlabel('Daily Average SYM-H Index')
#  if kind=='max':
  ax.set_xlabel('Ground Scatter Count')

#  ax.set_ylabel('(Hours of Observation) / 2')
#  ax.set_ylabel('Probability Density')
  ax.set_ylabel('Events')
  ax.set_title(title)
  ax.legend(loc='upper left')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_intpsd(dayList,radar):
  ''' 
  dayList is a list of datetime.datetime objects.
  
  Data should already be loaded into MongoDB using psd_load_manual.py.
  '''
  intpsdDict = {}
  for day in dayList:
    record = db['psd_manual'].find_one({'date':day,'radar':radar})

    if record == None:
      intpsdDict[day] = {'intpsd_sum': np.nan, 'intpsd_max': np.nan, 'intpsd_mean':np.nan}
    elif record['intpsd_sum'] == 'NaN':
      intpsdDict[day] = {'intpsd_sum': np.nan, 'intpsd_max': np.nan, 'intpsd_mean':np.nan}
    else:
      intpsdDict[day] = {'intpsd_sum': record['intpsd_sum'], 'intpsd_max': record['intpsd_max'], 'intpsd_mean':record['intpsd_mean']}
  return intpsdDict

def intpsd_histogram(gwDays,radar,secondPop=None,title='Integrated PSD Distribution',kind='mean',outFName='static/output/ground_scatter.png',max_range=None):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  intpsdLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    intpsdDict = get_intpsd(lst,radar)

    tmpLst = [intpsdDict[day]['intpsd_'+kind] for day in lst]
    intpsdLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)
    if max_range != None: mx = max_range
#    mx = 0.1
    rnge = [mn,mx]

    for intpsd in intpsdLst:
      n, bins = np.histogram(intpsd,bins=25,range=rnge,density=True)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
#    ax.set_xlim(left[0],0.1)
    ax.set_ylim(0,np.max(np.array(ys)))

  if kind=='mean':
    ax.set_xlabel('2 Hour Avg Integrated PSD')
  if kind=='max':
    ax.set_xlabel('2 Hour Max Integrated PSD')
  if kind=='sum':
    ax.set_xlabel('2 Hour Sum Integrated PSD')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Probability Density')
  ax.set_title(title)
  ax.legend(loc='upper right')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_ground_scatter_month(dayList,radar,sdt,fdt):
  ''' 
  dayList is a list of datetime.datetime objects.
  
  Data should already be loaded into MongoDB using psd_load_manual.py.
  '''
  #If a record for that radar and date range exists in the database, use it.
#  record = db['ground_scatter_month'].find_one({'radar':radar,'sdt':sdt,'fdt':fdt})
  record = None

  #If it does not exist, create the record.
  gscatMonthDict = {}
  if record == None:
    #Initialize the record dictionary
    record = {}
    record['radar'] = radar
    record['sdt']   = sdt
    record['fdt']   = fdt
    currentDate     = sdt
  
    #Create one entry in the record for every month
    for x in range(12):
      record[str(x+1)] = 0

    #Integrate the regular ground scatter databases into months.
    cursor  = db['ground_scatter'].find({'radar':radar,'date': {'$gte': sdt, '$lte': fdt}})
    tmpList = []
    if cursor != None:
      for x in cursor:
        if x['date'].hour < 14 or x['date'].hour > 20: continue
        tmpList.append(x['date'])
        print x['date'],x['gscat']
        if x['gscat'] != 'NaN':
          mn = str(x['date'].month)
          record[mn] = record[mn] + x['gscat']

#    db['ground_scatter_month'].insert(record)
  return record

  for day in dayList:
    record = db['psd_manual'].find_one({'date':day,'radar':radar})

    if record == None:
      intpsdDict[day] = {'intpsd_sum': np.nan, 'intpsd_max': np.nan, 'intpsd_mean':np.nan}
    elif record['intpsd_sum'] == 'NaN':
      intpsdDict[day] = {'intpsd_sum': np.nan, 'intpsd_max': np.nan, 'intpsd_mean':np.nan}
    else:
      intpsdDict[day] = {'intpsd_sum': record['intpsd_sum'], 'intpsd_max': record['intpsd_max'], 'intpsd_mean':record['intpsd_mean']}
  return intpsdDict

def ground_scatter_month_histogram(gwDays,radar,sdt,fdt,secondPop=None,title='Integrated PSD Distribution',kind='mean',outFName='static/output/ground_scatter.png',max_range=None):
  '''sdt: datetime.datetime start date
     fdt: datetime.datetime finish date'''
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(8,6))
  ax  = fig.add_subplot(111)

  dataLists = []
#  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  intpsdLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    gscatMonthDict = get_ground_scatter_month(lst,radar,sdt,fdt)
    tmpLst = [gscatMonthDict[str(x+1)] for x in range(12)]
#    import ipdb; ipdb.set_trace()
#    intpsdDict = get_intpsd(lst,radar)

#    tmpLst = [intpsdDict[day]['intpsd_'+kind] for day in lst]
    intpsdLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)
    if max_range != None: mx = max_range
#    mx = 0.1
    rnge = [mn,mx]

    for intpsd in intpsdLst:
#      n, bins = np.histogram(intpsd,bins=25,range=rnge,density=True)
      n = intpsd
      bins = np.arange(13) + 1

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
#    ax.set_xlim(left[0],0.1)
    ax.set_ylim(0,np.max(np.array(ys)))

  ax.set_xlabel('Month')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Count')
#  ax.set_title(title)
  ax.set_title('Ground Scatter Counts between 1400-2000 UT\n'+title)
#  ax.legend(loc='upper right')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def month_histogram(gwDays,radar,secondPop=None,title='Month Distribution',kind='mean',outFName='static/output/month_distribution.png',max_range=None):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  intpsdLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    tmpLst = [day.month for day in lst]

#    intpsdDict = get_intpsd(lst,radar)
#
#    tmpLst = [intpsdDict[day]['intpsd_'+kind] for day in lst]
    intpsdLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)
    if max_range != None: mx = max_range
#    mx = 0.1
    rnge = [mn,mx]

    for intpsd in intpsdLst:
#      n, bins = np.histogram(intpsd,bins=12,range=[1,12],density=True)
      n, bins = np.histogram(intpsd,bins=12,range=[1,12],density=False)
      bins = np.arange(13)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = lbl_0
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = lbl_1
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
#    ax.set_xlim(left[0],0.1)
    ax.set_ylim(0,np.max(np.array(ys)))

  ax.set_xlabel('Month')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Events')
  ax.set_title(title)
  ax.legend(loc='upper left')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def month_histogram2(gwDays,radar,secondPop=None,title='Month Distribution',kind='mean',outFName='static/output/month_distribution.png',max_range=None):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  intpsdLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    tmpLst = [day.month for day in lst]

#    intpsdDict = get_intpsd(lst,radar)
#
#    tmpLst = [intpsdDict[day]['intpsd_'+kind] for day in lst]
    intpsdLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))
  
  binLst = []
  for item in dataLists:
    tmp = {}
    for nr in range(12):
      tmp[nr+1] = 0

    for iitem in item:
      month = iitem.month
      tmp[month] = tmp[month] + 1

    binLst.append(tmp)

  normLst  = []
  totalLst  = []
  for inx in range(12):
    totalTmp = binLst[0][inx+1]+binLst[1][inx+1]
    if totalTmp != 0:
      normTmp  = binLst[1][inx+1] / float(totalTmp)
    else:
      normTmp = 0

    totalLst.append(totalTmp)
    normLst.append(normTmp)

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)
    if max_range != None: mx = max_range
#    mx = 0.1
    rnge = [mn,mx]

#    for intpsd in intpsdLst:
    intpsd = intpsdLst[0]
#      n, bins = np.histogram(intpsd,bins=12,range=[1,12],density=True)
#    n, bins = np.histogram(intpsd,bins=12,range=[1,12],density=False)
    n = normLst
    bins = np.arange(13)

    # get the corners of the rectangles for the histogram
    left    = np.array(bins[:-1])
    right   = np.array(bins[1:])
    bottom  = np.zeros(len(left))
    top     = bottom + n

    xs.append(left)
    xs.append(right)
    ys.append(top)

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    if inx == 0 and len(dataLists) == 2:
      clr   = clr_0
      lbl   = lbl_0
      alpha = alpha_0
    else:
      clr   = clr_1
      lbl   = lbl_1
      alpha = alpha_1
    patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
    ax.add_patch(patch)
    inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
#    ax.set_xlim(left[0],0.1)
    ax.set_ylim(0,np.max(np.array(ys)))

  ax.set_xlabel('Month')

#  ax.set_ylabel('(Hours of Observation) / 2')
  ax.set_ylabel('Events')
  ax.set_title(title)
  ax.legend(loc='upper left')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')
def ground_scatter_discard_histogram(gwDays,radar,secondPop=None,title='Ground Scatter Distribution',kind='mean',outFName='static/output/ground_scatter.png',max_range=None):
  import matplotlib.path as path
  import matplotlib.patches as patches

  fig = Figure(figsize=(6,6))
  ax  = fig.add_subplot(111)

  dataLists = []
  if secondPop != None: dataLists.append(secondPop)
  dataLists.append(gwDays)

  inx     = 0
  ys      = []
  xs      = []
  gsLst   = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    gsDict = get_ground_scatter(lst,radar)

    tmpLst = [gsDict[day]['gscat'] for day in lst]
    gsLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)

    if max_range != None: mx = max_range
    rnge = [mn,mx]

    for gs in gsLst:
      n, bins = np.histogram(gs,bins=25,range=rnge,density=False)

      # get the corners of the rectangles for the histogram
      left    = np.array(bins[:-1])
      right   = np.array(bins[1:])
      bottom  = np.zeros(len(left))
      top     = bottom + n

      xs.append(left)
      xs.append(right)
      ys.append(top)

      # we need a (numrects x numsides x 2) numpy array for the path helper
      # function to build a compound path
      XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T

      # get the Path object
      barpath = path.Path.make_compound_path_from_polys(XY)

      # make a patch out of it
      if inx == 0 and len(dataLists) == 2:
        clr   = clr_0
        lbl   = 'Discard Observations'
        alpha = alpha_0
      else:
        clr   = clr_1
        lbl   = 'Accepted Observations'
        alpha = alpha_1
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=alpha,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
#    ax.set_xlim(left[0], right[-1])
    ax.set_xlim(0, right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

#  if kind=='mean':
#    ax.set_xlabel('Daily Average SYM-H Index')
#  if kind=='max':
  ax.set_xlabel('Ground Scatter Count')

#  ax.set_ylabel('(Hours of Observation) / 2')
#  ax.set_ylabel('Probability Density')
  ax.set_ylabel('Events')
  ax.set_title(title)
  ax.legend(loc='upper right')

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')
