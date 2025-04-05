#!/usr/bin/env python
#From http://flask.pocoo.org/docs/tutorial/setup/#tutorial-setup
#all the imports
# from flask import Flask, request, session, redirect, url_for, abort, render_template, flash, jsonify
import jsonify
import flash
import pymongo
# from bson.objectid import ObjectId

import datetime
import os
# import sys

import numpy as np
# import glob
# from scipy.io.idl import readsav

# from scipy import signal

# from davitpy import utils

# from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

mongo         = pymongo.MongoClient()
db            = mongo.mstid

def days(selected=None):
  days = []
  for dd in range(31):
    day = str(dd+1)
    if dd+1 == selected:
      sel = True
    else:
      sel = False
    days.append({'text':day,'value':day,'selected':sel})
  return days

def months(selected=None):
  mon_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  months = []
  for dd in range(12):
    month = str(dd+1)
    if dd+1 == selected:
      sel = True
    else:
      sel = False
    months.append({'text':mon_list[dd],'value':month,'selected':sel})
  return months

def years(selected=None):
  years = []
  yrs = list(range(1993,datetime.datetime.utcnow().year+1))
  for dd in yrs:
    year = str(dd)
    if dd == selected:
      sel = True
    else:
      sel = False
    years.append({'text':year,'value':year,'selected':sel})
  return years

def db_update_mstid_list(item,mstid_list='mstid_list'):
  if '_id' in item: item.pop('_id')
  entry_id = db[mstid_list].update_one({'date':item['date'], 'radar':item['radar']}, {"$set": item}, upsert=True)
  return entry_id

def linkUp(dayList):
    dayStr = []
    for x in dayList:

        img1 = 'x'
        img2 = 'x'

        if 'category_manu' in x:
          img1 = 'check'
          if 'checked' in x:
              if x['checked']:
                img2 = 'check'
#          if x['category_auto'] == x['category_manu']:
#            img2 = 'check'

        sz        = '10'
        xstr      = x['date'].strftime('%Y%m%d-%H%M') 
        disp_str  = x['radar']+x['date'].strftime('.%Y%m%d-%H%M') 
        anc1_tag  = '<a class="btn btn-primary mt-2" onclick="plotRTI(\''+x['radar']+'\',\''+xstr+'\',\'power\');">'
        # <img src="{% static 'images/rad1.png' %}" class="d-block h-50 w-100 " alt="...">
        img1_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/staticfiles/images/'+img1+'.png">'
        img2_tag  = '<img width="'+sz+'px" height="'+sz+'px" src="/staticfiles/images/'+img2+'.png">'
        anc2_tag  = '</a>'

        fstr      = ''.join([anc1_tag,img1_tag,img2_tag,disp_str,anc2_tag])

        dayStr.append(fstr)

    dayStr = '<br />\n'.join(dayStr)
    return dayStr

def kpBins(width=0.1):
  kpCtr = [0.,0.33,0.67,1.,1.33,1.67,2.,2.33,2.67,3.,3.33,3.67,4.,4.33,4.67,5.,5.33,5.67,6.,6.33,6.67,7.,7.33,7.67,8.,8.33,8.67,9]

  bins = []
  for kp in kpCtr:
    bins.append(kp-width)
    bins.append(kp+width)
  return bins
    
def get_KP_day(dayList):
  '''Download KP data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the KP dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  kpDict = {}
  for day in dayList:
    record = db['kpDay'].find_one({'date':day})
    if record == None:
      eTime = day+datetime.timedelta(hours=24)
#      kpList = gme.ind.kp.readKp(sTime=day, eTime=eTime)
      kpList = gme.ind.kp.readKp(sTime=day,eTime=day)
      oneDayList = []
      for kp in kpList:
        if kp.time != day: continue
        ap_mean = kp.apMean
        ap_max  = max(kp.ap)
        kp_mean = apToKp(ap_mean)
        kp_max  = apToKp(ap_max)
      record = {'date':day,'ap_mean':ap_mean,'ap_max':ap_max,'kp_mean':kp_mean, 'kp_max':kp_max}
      db['kpDay'].insert(record)
    kpDict[day] = {'ap_mean':record['ap_mean'], 'ap_max':record['ap_max'],'kp_mean':record['kp_mean'], 'kp_max':record['kp_max']}
  return kpDict

def kpHistogram(gwDays,secondPop=None,title='Kp Distribution',kind='mean',outFName='staticfiles/output/kp.png'):
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
  lng = []
  for lst in dataLists:
    if lst==[]: continue
    sTime = min(lst)
    eTime = max(lst)

    kpDict = get_KP_day(lst)

    kp = [kpDict[day]['kp_'+kind] for day in lst]

    n, bins = np.histogram(kp, kpBins(0.15))
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
      clr = 'green'
      lbl = 'Quiet Days'
    else:
      clr = 'blue'
      lbl = 'MSTID Days'
    patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=0.8,label=lbl)
    ax.add_patch(patch)
    inx = inx+1

  # update the view limits
#  ax.set_xlim(left[0], right[-1])
  ax.set_xlim(-0.20, 9.2)
  if ys != []:
    ax.set_ylim(0,np.max(np.array(ys)))
  if kind=='mean':
    ax.set_xlabel('Daily Average Kp Index\n(Average from Ap Index)')
  if kind=='max':
    ax.set_xlabel('Daily Maximum AE Index')
  ax.set_ylabel('Number Days')
  ax.set_xticks(list(range(10)))
  ax.set_title(title)
  ax.legend()

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_AE_day(dayList):
  '''Download AE data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the AE dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  aeDict = {}
  for day in dayList:
    record = db['aeDay'].find_one({'date':day})
    if record == None:
      eTime = day+datetime.timedelta(hours=24)
      aeList = gme.ind.ae.readAe(sTime=day, eTime=eTime)
      oneDayList = []
      for ae in aeList:
        if ae.time >= day and ae.time < eTime: oneDayList.append(ae.ae)
      record = {'date':day,'mean':np.mean(oneDayList),'max':np.max(oneDayList)}
      db['aeDay'].insert(record)
    aeDict[day] = {'mean':record['mean'], 'max':record['max']}
  return aeDict
    
def aeHistogram(gwDays,secondPop=None,title='AE Distribution',kind='mean',outFName='staticfiles/output/ae.png'):
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
  lng     = []
  aeLst   = []
  maxLst  = []
  for lst in dataLists:
    if lst==[]: continue
    sTime = min(lst)
    eTime = max(lst)

    aeDict = get_AE_day(lst)

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
      n, bins = np.histogram(ae,bins=25,range=rnge)

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
        clr = 'green'
        lbl = 'Quiet Days'
      else:
        clr = 'blue'
        lbl = 'MSTID Days'
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=0.8,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

  if kind=='mean':
    ax.set_xlabel('Daily Average AE Index')
  if kind=='max':
    ax.set_xlabel('Daily Maximum AE Index')

  ax.set_ylabel('Number Days')
#  ax.set_xticks(range(10))
  ax.set_title(title)
  ax.legend()

  canvas = FigureCanvasAgg(fig)
  canvas.print_figure(outFName,format='png',facecolor='white',edgecolor='white')

def get_SYMH_day(dayList):
  '''Download SYMH data and calculate an average and a maximum for each day in the given dayList.
  Results are stored to the mongo database.  If the record is already available in the mongoDb, 
  then it is not downloaded again, and the SYMH dictionaries from mongoDB are returned.
  
  dayList is a list of datetime.datetime objects.
  '''
  import gme
  symhDict = {}
  for day in dayList:
    record = db['symhDay'].find_one({'date':day})
    if record == None:
      eTime = day+datetime.timedelta(hours=24)
      symhList = gme.ind.symasy.readSymAsy(sTime=day, eTime=eTime)

      if symhList == None:
        record = {'date':day,'mean':np.nan,'max':np.nan}
      else:
        oneDayList = []
        for symh in symhList:
          if symh.time >= day and symh.time < eTime: oneDayList.append(symh.symh)
        record = {'date':day,'mean':np.mean(oneDayList),'max':np.max(oneDayList)}
      db['symhDay'].insert(record)

    symhDict[day] = {'mean':record['mean'], 'max':record['max']}
  return symhDict
    
def symhHistogram(gwDays,secondPop=None,title='SYM-H Distribution',kind='mean',outFName='staticfiles/output/symh.png'):
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
  lng     = []
  symhLst = []
  maxLst  = []
  minLst  = []
  for lst in dataLists:
    if lst == []: continue
    sTime = min(lst)
    eTime = max(lst)

    symhDict = get_SYMH_day(lst)

    tmpLst = [symhDict[day][kind] for day in lst]
    symhLst.append(tmpLst)
    maxLst.append(max(tmpLst))
    minLst.append(min(tmpLst))

  if maxLst != []:
    mx = max(maxLst)
    mn = min(minLst)

    rnge = [mn,mx]

    for symh in symhLst:
      n, bins = np.histogram(symh,bins=25,range=rnge)

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
        clr = 'green'
        lbl = 'Quiet Days'
      else:
        clr = 'blue'
        lbl = 'MSTID Days'
      patch = patches.PathPatch(barpath, facecolor=clr, edgecolor='gray', alpha=0.8,label=lbl)
      ax.add_patch(patch)
      inx = inx+1

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(0,np.max(np.array(ys)))

  if kind=='mean':
    ax.set_xlabel('Daily Average SYM-H Index')
  if kind=='max':
    ax.set_xlabel('Daily Maximum SYM-H Index')

  ax.set_ylabel('Number Days')
  ax.set_title(title)
  ax.legend()

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

def listDropDown():
    active_list = get_active_list()

    #Load in items from listTracker collection and remove any that do not have valid references.
#    items  = [x for x in db['listTracker'].find().sort('name',1)]
    
    items   = []
    for x in db['listTracker'].find().sort('name',1):
        if x['name'] not in db.list_collection_names():
            entry = db['listTracker'].find_one_and_delete({"_id":x['_id']})
        else:
            items.append(x)


    html = []
    html.append('<select class="re-size form-select form-select-lg" name="list_dropdown" id="list_dropdown" style="display: inline; width: 80%;">')
#    html.append('  <option value=""></option>')
#    html.append('  <option value="clean">--Clean List--</option>')
    html.append('  <option value="saveAs">Save List As...</option>')
    for x in items:
        value = x['name']
        text  = x['name']
        if value == active_list:
          selected=' selected="selected"'
        else:
          selected=''
        try:
            tag = '  <option value="'+value+'"'+selected+'>'+text+'</option>'
        except:
            tag = '  <option value="ERROR">ERROR</option>'
        html.append(tag)
    html.append('</select>')
    html.append('<button id="listLoadButton" class="button-62" role="button" style="position:relative; left: .2rem; right: 0;">Load</button>')
#    html.append('<button id="listSaveButton">Save</button>')
    html.append('<button id="listDeleteButton" class="button-62" role="button" style="position:relative; left:.2rem; right: 0;">Delete</button>')
    html = '\n'.join(html)
    return html

def set_active_list(name):
    '''Set a mongodb _id to the actively active list.'''
    db.active_list.delete_many({})
    tmp = db.active_list.insert_one({'name':name})

def get_active_list():
    '''Get the active list and create new ones if there are none.'''
    active_list = db['active_list'].find_one()
    if active_list != None:
        list_name     = active_list['name']
    else:
        list_name     = 'default_list'

    test = list_name in db.list_collection_names()
    if test == None:
        active_list = db['listTracker'].find_one()
        list_name   = active_list['name']
        set_active_list(list_name)

    return list_name

def set_nav_mode(nav_mode):
    db['settings'].delete_one({'nav_mode':{'$exists':True}})
    db['settings'].insert_one({'nav_mode':nav_mode})
    return

def get_nav_mode():
    '''Get the active navigation mode and set one if there is none.'''
    item = db['settings'].find_one({'nav_mode': {'$exists': True}})
    if item is None:
        nav_mode = 'list'
        db['settings'].insert_one({'nav_mode':nav_mode})
    else:
        nav_mode = str(item['nav_mode'])

    return nav_mode

def loadDayLists(mstid_list='mstid_list',gs_sort=False):
    '''Load the MSTID, Quiet, and None day lists from the database.'''
    ################################################################################
    #Load from database
    if gs_sort == False:
        sort_term  = 'date'
        sort_order = 1
    else:
        sort_term  = 'gscat'
        sort_order = -1

    def getDB(category):
        if category == 'unclassified':
            cursor = db[mstid_list].find({'category_manu':{'$exists':False}}).sort(sort_term,sort_order)
        else:
            cursor = db[mstid_list].find({'category_manu':category}).sort(sort_term,sort_order)
#    cursor = db[mstid_list].find(
#      {'$or': [
#        {'category_manu': {'$exists': True},  'category_manu':category},
#        {'category_manu': {'$exists': False}, 'category_auto':category}]
#        }).sort(sort_term,sort_order)
        dbDict = [x for x in cursor]
        return dbDict

    mstidDays  = getDB('mstid')
    quietDays  = getDB('quiet')
    noneDays   = getDB('None')
    unclassifiedDays   = getDB('unclassified')
      
    return (mstidDays,quietDays,noneDays,unclassifiedDays)
