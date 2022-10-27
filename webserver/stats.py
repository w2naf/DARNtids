#!/usr/bin/env python
#From http://flask.pocoo.org/docs/tutorial/setup/#tutorial-setup
#all the imports
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, session, redirect, url_for, abort, render_template, flash, jsonify
import pymongo
from bson.objectid import ObjectId

import datetime
import os
import shutil
import sys
import pickle

import numpy as np
import glob
from scipy.io.idl import readsav

from scipy import signal

from davitpy import utils

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from davitpy import pydarn

#Confuguration
DEBUG = True
SECRET_KEY = 'HJJDnaoiwer&*(@#%@sdanbiuas@HEIu'
USERNAME = 'admin'
PASSWORD = 'default'

import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

app = Flask(__name__)
app.config.from_object(__name__)

#sys.path.append(os.path.join(app.root_path,'pyprop'))

mongo         = pymongo.MongoClient()
db            = mongo.mstid

app.config.from_envvar('FLASKR_SETTINGS',silent=True)

from stats_support import *

import manual_support as mans
import music_support as msc

@app.route('/')
@app.route('/manual')
def manual_search():
    mstid_list  = get_active_list()

    mstidDayDict,quietDayDict,noneDayDict,unclassifiedDayDict = loadDayLists(mstid_list=mstid_list)

    mstidStr  = linkUp(mstidDayDict)
    quietStr  = linkUp(quietDayDict)
    noneStr   = linkUp(noneDayDict)
    unclassifiedStr   = linkUp(unclassifiedDayDict)

    webData = {}
    webData['mstidDays']      = mstidStr
    webData['quietDays']      = quietStr
    webData['noneDays']       = noneStr
    webData['unclassifiedDays'] = unclassifiedStr 

    # Count number of events. ######################################################
    webData['mstid_days_total'] = len(mstidDayDict)
    mstid_days_manual_checked = 0
    for event in mstidDayDict:
      if event.has_key('category_manu'):
          mstid_days_manual_checked = mstid_days_manual_checked + 1
    webData['mstid_days_manual_checked'] = mstid_days_manual_checked

    webData['quiet_days_total'] = len(quietDayDict)
    quiet_days_manual_checked = 0
    for event in quietDayDict:
      if event.has_key('category_manu'):
          quiet_days_manual_checked = quiet_days_manual_checked + 1
    webData['quiet_days_manual_checked'] = quiet_days_manual_checked

    webData['none_days_total'] = len(noneDayDict)
    none_days_manual_checked = 0
    for event in noneDayDict:
      if event.has_key('category_manu'):
          none_days_manual_checked = none_days_manual_checked + 1
    webData['none_days_manual_checked'] = none_days_manual_checked

    webData['unclassified_days_total'] = len(unclassifiedDayDict)
    unclassified_days_manual_checked = 0
    for event in unclassifiedDayDict:
      if event.has_key('category_manu'):
          unclassified_days_manual_checked = unclassified_days_manual_checked + 1
    webData['unclassified_days_manual_checked'] = unclassified_days_manual_checked

    webData['days_total']       = webData['mstid_days_total'] + webData['quiet_days_total'] + webData['none_days_total'] + webData['unclassified_days_total']
    ################################################################################

    webData['list_dropdown']    = listDropDown()
    webData['mstid_list']       = mstid_list
    webData['homeURL']          = '/'

    timestamp=datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    return render_template('manual.html',webData=webData,timestamp=timestamp)

@app.route('/select_source', methods=['GET'])
def select_source():
    result = 1
    new_source    = request.args.get('sources_dropdown',None,type=str)
    source_link   = msc.get_output_path()
    if os.path.islink(source_link):
        os.remove(source_link)
        os.symlink(new_source,source_link)
  
        flash('Data source selected: {}'.format(new_source))
        result=0

    return jsonify(result=result)

@app.route('/list_save_as',methods=['GET'])
def list_save_as():
    list_name   = request.args.get('listName',None,type=str)
    saveAsName  = request.args.get('saveAsName',None,type=str)
    mstid_list  = request.args.get('mstid_list',None,type=str)
       
    if list_name == 'saveAs':
        saveAs    = True
        list_name  = saveAsName
    else:
        saveAs    = False

    checkName = list_name in db.collection_names()
    if list_name == '' or list_name == 'clean':
        result = 1
    elif checkName == True and saveAs == True:
        flash('Error! A collection with the name "'+list_name+'" already exists!')
        result = 1
    else:
        if checkName == False:
            db['listTracker'].insert({'name':list_name})
            for x in db[mstid_list].find():
                db[list_name].insert(x)
            set_active_list(list_name)
            result = 0
    return jsonify(result=result)

@app.route('/load_list', methods=['GET'])
def load_list():
  listName    = request.args.get('list_dropdown',None,type=str)
  mstid_list  = request.args.get('mstid_list',None,type=str)
  result = 1
  if listName != '' and listName != 'saveAs':
    set_active_list(listName)
    flash('List "'+listName+'" loaded.')
    result=0
  return jsonify(result=result)

@app.route('/list_delete', methods=['GET'])
def list_delete():
    listName    = request.args.get('list_dropdown',None,type=str)
    mstid_list  = request.args.get('mstid_list',None,type=str)
    result = 1
    if listName != '' and listName != 'saveAs' and listName != None and listName != 'clean':
        entry = db['listTracker'].find_and_modify({"name":listName},remove=True)
        db[listName].drop()
        flash('List "'+listName+'" deleted.')
        result=0
    return jsonify(result=result)

@app.route('/update_category',methods=['GET'])
def update_category():
    '''Update the categories datebase for a particular day.'''

    #Unpack variables from the post.
    mstid_list                = request.args.get('mstid_list',None,type=str)
    settings_collection_name  = request.args.get('settings_collection_name',None,type=str)

    item                  = {}
    item['radar']         = request.args.get('categ_radar', 0, type=str)
    item['date']          = datetime.datetime.strptime(request.args.get('categ_day', 0, type=str),'%Y%m%d-%H%M')

    category_manu = request.args.get('categ_manu', 'Null', type=str)
    if category_manu != 'Null':
        if category_manu != 'None':
            item['category_manu'] = category_manu
        else:
            item['category_manu'] = 'None'

    checked = request.args.get('categ_checked')
    if checked == 'true':
        item['checked'] = True
    else:
        item['checked'] = False

#    item['notes']         = request.args.get('categ_notes', 0, type=str)

    tmp = db_update_mstid_list(item,mstid_list=mstid_list)
    radar = item['radar']
    mstidDayDict,quietDayDict,noneDayDict,unclassifiedDayDict = loadDayLists(mstid_list=mstid_list)

    mstidStr  = linkUp(mstidDayDict)
    quietStr  = linkUp(quietDayDict)
    noneStr   = linkUp(noneDayDict)
    unclassifiedStr   = linkUp(unclassifiedDayDict)

    return jsonify(mstidStr=mstidStr,quietStr=quietStr,noneStr=noneStr)

@app.route('/music_update_category',methods=['GET'])
def music_update_category():
    '''Update the categories datebase for a particular day.
       This version is called from the music_edit page.'''

    #Unpack variables from the post.
    mstid_list      = request.args.get('mstid_list',None,type=str)
    category_manu   = request.args.get('categ_manu', 'None', type=str)
    str_id          = request.args.get('_id',None,type=str)

    _id = ObjectId(str_id)
    event   = db[mstid_list].find_one({'_id':_id})

    status = db[mstid_list].update({'_id':_id},{'$set': {'category_manu':category_manu}})

    if not status['err']:
        result=0
    else:
        result=1

    nav_mode = get_nav_mode()
    prev_url,next_url  = msc.get_prev_next(mstid_list,_id,mode=nav_mode)
    urls= {}
    urls['prev_url'] = prev_url
    urls['next_url'] = next_url

    return jsonify(result=result,category_manu=category_manu,urls=urls)

@app.route('/update_nav_mode',methods=['GET'])
def update_nav_mode():

    #Unpack variables from the post.
    nav_mode        = request.args.get('nav_mode','list',type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    _id = ObjectId(str_id)

    set_nav_mode(nav_mode)

    urls = msc.get_prev_next(mstid_list,_id,mode=nav_mode)
    result = {}
    result['prev_url'] = urls[0]
    result['next_url'] = urls[1]
    
    return jsonify(result=result)

# Generate RTI Plot ############################################################
@app.route('/rti',methods=['GET'])
def plot_rti():
  '''Plot an RTI plot for the given day to a PNG and return the PNG's location and
  some information about that day from the database.'''

  #Unpack variables from the get.
  radar       = request.args.get('radar', 0, type=str)
  gwDay       = request.args.get('gwDay', 0, type=str)
  param       = request.args.get('param', 0, type=str)
  mstid_list  = request.args.get('mstid_list', None, type=str)

  stime   = datetime.datetime.strptime(gwDay,'%Y%m%d-%H%M')
  shortDt = datetime.datetime.strptime(gwDay[0:8],'%Y%m%d')

  #Build the path of the RTI plot and call the plotting routine.
  d = 'static/rti'
  outputFile = d+'/'+stime.strftime('%Y%m%d')+'.'+radar+'.'+param+'rti.png'
  try:
    os.makedirs(d)
  except:
    pass

  if not os.path.exists(outputFile):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)
    tick_size   = 16

    xticks  = []
    hours   = np.arange(0,26,2) #Put a tick every 2 hours.
    for hour in hours:
        tmp = shortDt + datetime.timedelta(hours=float(hour))
        xticks.append(tmp)
    axvlines  = xticks[:]
    fig = Figure(figsize=(20,10))
    fig = pydarn.plotting.rti.plotRti(shortDt,radar,params=['power'],show=False,retfig=True,figure=fig,xtick_size=tick_size,ytick_size=tick_size,xticks=xticks,axvlines=axvlines)
    canvas = FigureCanvasAgg(fig)
    fig.savefig(outputFile)

  #Load in infomation about day stored in database.
  dbDict = db[mstid_list].find_one({'radar':radar,'date':stime})
  if dbDict == None: dbDict = {}

  #Pack everything up into a dictionary to return to the JavaScript function.
  output            = {}
  #Append timestamp to force reload of image.
  output['result']  = outputFile+datetime.datetime.now().strftime('?%Y%m%d%H%M%S')
  output['radar']   = radar
  output['gwDay']   = gwDay

  output['categ_auto_mstid'] = False
  output['categ_auto_quiet'] = False
  output['categ_auto_none']  = False
  if dbDict.has_key('category_auto'):
    if dbDict['category_auto'] == 'mstid':
      output['categ_auto_mstid'] = True
    elif dbDict['category_auto'] == 'quiet':
      output['categ_auto_quiet'] = True
    elif dbDict['category_auto'] == 'None':
      output['categ_auto_none'] = True

  output['categ_manu_mstid'] = False
  output['categ_manu_quiet'] = False
  output['categ_manu_none']  = False
  if dbDict.has_key('category_manu'):
    if dbDict['category_manu'] == 'mstid':
      output['categ_manu_mstid'] = True
    elif dbDict['category_manu'] == 'quiet':
      output['categ_manu_quiet'] = True
    elif dbDict['category_manu'] == 'None':
      output['categ_manu_none'] = True

  if dbDict.has_key('checked'):
    output['categ_checked'] = dbDict['checked']
  else:
    output['categ_checked'] = None

  if dbDict.has_key('survey_code'):
    output['survey_code'] = dbDict['survey_code']
  else:
    output['survey_code'] = None

  if dbDict.has_key('mlt'):
    mlt = dbDict['mlt']
    hr  = int(mlt) * 100.
    mn  = (mlt%1) * 60.
    mlt_str = '%04.0f' % (hr+mn)
    output['mlt'] = mlt_str
  else:
    output['mlt'] = None

  if dbDict.has_key('gscat'):
    output['gscat'] = dbDict['gscat']
  else:
    output['gscat'] = None

  if dbDict.has_key('notes'):
    output['categ_notes'] = dbDict['notes']
  else:
    output['categ_notes'] = None
  return jsonify(**output)

@app.route('/music')
def music():
    mstid_list  = get_active_list()

    mstidDayDict,quietDayDict,noneDayDict,unclassifiedDayDict = loadDayLists(mstid_list=mstid_list)

    mstidStr  = msc.linkUp(mstidDayDict)
    quietStr  = msc.linkUp(quietDayDict)
    noneStr   = msc.linkUp(noneDayDict)
    unclassifiedStr   = msc.linkUp(unclassifiedDayDict)

    webData = {}

    webData['mstidDays']      = mstidStr
    webData['quietDays']      = quietStr
    webData['noneDays']       = noneStr
    webData['unclassifiedDays'] = unclassifiedStr 

    # Count number of events. ######################################################
    webData['mstid_days_total'] = len(mstidDayDict)
    mstid_days_manual_checked = 0
    for event in mstidDayDict:
      if event.has_key('category_manu'):
          mstid_days_manual_checked = mstid_days_manual_checked + 1
    webData['mstid_days_manual_checked'] = mstid_days_manual_checked

    webData['quiet_days_total'] = len(quietDayDict)
    quiet_days_manual_checked = 0
    for event in quietDayDict:
      if event.has_key('category_manu'):
          quiet_days_manual_checked = quiet_days_manual_checked + 1
    webData['quiet_days_manual_checked'] = quiet_days_manual_checked

    webData['none_days_total'] = len(noneDayDict)
    none_days_manual_checked = 0
    for event in noneDayDict:
      if event.has_key('category_manu'):
          none_days_manual_checked = none_days_manual_checked + 1
    webData['none_days_manual_checked'] = none_days_manual_checked

    webData['unclassified_days_total'] = len(unclassifiedDayDict)
    unclassified_days_manual_checked = 0
    for event in unclassifiedDayDict:
      if event.has_key('category_manu'):
          unclassified_days_manual_checked = unclassified_days_manual_checked + 1
    webData['unclassified_days_manual_checked'] = unclassified_days_manual_checked

    webData['days_total']       = webData['mstid_days_total'] + webData['quiet_days_total'] + webData['none_days_total'] + webData['unclassified_days_total']
    ################################################################################

    # Aggregate things to get out important date info. #############################
    totalDayDictList    = mstidDayDict + quietDayDict + noneDayDict + unclassifiedDayDict
    totalMLTList        = [x['mlt'] for x in totalDayDictList]
    mstidMLTList        = [x['mlt'] for x in mstidDayDict]
    totalMLTArr         = np.array(totalMLTList)
    mstidMLTArr         = np.array(mstidMLTList)

    webData['list_dropdown']    = listDropDown()
    webData['mstid_list']       = mstid_list
    webData['homeURL']          = '/music'
    
    try:
        webData['min_mlt']  = '{0:.2f}'.format(totalMLTArr.min())
        webData['max_mlt']  = '{0:.2f}'.format(totalMLTArr.max())
    except:
        pass

    webData['nr_mstid_08_10MLT']    = '{0:d}'.format(np.sum(np.logical_and(mstidMLTArr >=  8., mstidMLTArr < 10.)))
    webData['nr_mstid_10_12MLT']    = '{0:d}'.format(np.sum(np.logical_and(mstidMLTArr >= 10., mstidMLTArr < 12.)))
    webData['nr_mstid_12_14MLT']    = '{0:d}'.format(np.sum(np.logical_and(mstidMLTArr >= 12., mstidMLTArr < 14.)))
    webData['nr_mstid_14_16MLT']    = '{0:d}'.format(np.sum(np.logical_and(mstidMLTArr >= 14., mstidMLTArr < 16.)))
    webData['nr_mstid_16_18MLT']    = '{0:d}'.format(np.sum(np.logical_and(mstidMLTArr >= 16., mstidMLTArr < 18.)))

    messages    = []
    if len(messages) > 0:
        webData['messages_on'] = True
    else:
        webData['messages_on'] = False
    webData['messages']         = messages

    webData['source_selector']      = msc.sourcesDropDown()
    
    timestamp=datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    tmp = url_for('music_edit',name='name')
    return render_template('music.html',webData=webData,timestamp=timestamp)

@app.route('/music_edit',methods=['GET'])
def music_edit():
    '''Plot an RTI plot for the given day to a PNG and return the PNG's location and
    some information about that day from the database.'''

    mstid_list  = get_active_list()

    #Define some dictionaries to be sent to the web template,
    params  = {}
    webData = {}
    musicParams = {}

    #Unpack variables from the get.
    radar       = request.args.get('radar', 0, type=str).lower()
    sDate       = request.args.get('sDate', 0, type=str)
    fDate       = request.args.get('fDate', 0, type=str)
    _idStr      = request.args.get('_id', 0, type=str)

    _id         = ObjectId(_idStr)
    rec         = db[mstid_list].find_one({'_id': _id})
    #  param       = request.args.get('param', 0, type=str)
    #  mstid_list  = request.args.get('mstid_list', None, type=str)


    sDatetime = datetime.datetime.strptime(sDate,'%Y%m%d.%H%M') 
    fDatetime = datetime.datetime.strptime(fDate,'%Y%m%d.%H%M')

    #Build up list of everything in database record to make it easy to print
    #everything out in Jinja2.
    record_list = []
    keys = rec.keys()
    keys.sort()
    for key in keys:
        record_list.append({'key':key,'value':rec[key]})

    musicParams_list = False    #Don't show runfile list in web if there is no run file.
    try:
        runFile     = msc.load_runfile(radar,sDatetime,fDatetime)
        musicParams = runFile.runParams

        #List for Jinja2 dump.
        musicParams_list = []
        keys = musicParams.keys()
        keys.sort()
        for key in keys:
            musicParams_list.append({'key':key,'value':musicParams[key]})

    except:
        musicParams['radar']                = radar
        musicParams['sDatetime']            = sDatetime
        musicParams['fDatetime']            = fDatetime

        musicParams['beamLimits']           = None
        musicParams['gateLimits']           = None
        musicParams['interpRes']            = 120
        musicParams['filter_numtaps']       = 101
        musicParams['filter_cutoff_low']    = 0.0003
        musicParams['filter_cutoff_high']   = 0.0012
        musicParams['kx_max']               = 0.05
        musicParams['ky_max']               = 0.05
        musicParams['autodetect_threshold'] = 0.35
        musicParams['neighborhood']         = (10, 10)

    if musicParams.has_key('beamLimits'):
        if np.size(musicParams['beamLimits']) == 2:
            musicParams['beamLimits_0'] = musicParams['beamLimits'][0]
            musicParams['beamLimits_1'] = musicParams['beamLimits'][1]
        else:
            musicParams['beamLimits_0'] = None
            musicParams['beamLimits_1'] = None
    else:
        musicParams['beamLimits_0'] = None
        musicParams['beamLimits_1'] = None

    if musicParams.has_key('gateLimits'):
        if np.size(musicParams['gateLimits']) == 2:
            musicParams['gateLimits_0'] = musicParams['gateLimits'][0]
            musicParams['gateLimits_1'] = musicParams['gateLimits'][1]
        else:
            musicParams['gateLimits_0'] = None
            musicParams['gateLimits_1'] = None
    else:
        musicParams['gateLimits_0'] = None
        musicParams['gateLimits_1'] = None

    if musicParams.has_key('neighborhood'):
        if np.size(musicParams['neighborhood']) == 2:
            musicParams['neighborhood_0'] = musicParams['neighborhood'][0]
            musicParams['neighborhood_1'] = musicParams['neighborhood'][1]
        else:
            musicParams['neighborhood_0'] = 10
            musicParams['neighborhood_1'] = 10
    else:
        musicParams['neighborhood_0'] = 10
        musicParams['neighborhood_1'] = 10

    sTime   = musicParams.get('sDatetime',musicParams.get('sTime'))
    eTime   = musicParams.get('fDatetime',musicParams.get('eTime'))

    musicPath   = msc.get_output_path(musicParams['radar'],sTime,eTime)
    pickleName  = msc.get_pickle_name(musicParams['radar'],sTime,eTime)
    picklePath  = os.path.join(musicPath,pickleName)

    dataObj = None
    try:
        with open(picklePath,'rb') as fl:
            dataObj     = pickle.load(fl)
            webData['musicObjStatusClass']  = 'statusNormal'
            webData['musicObjStatus']       = 'Using musicObj file < '+ picklePath +' >.'
    except:
            webData['musicObjStatusClass']  = 'warning'
            webData['musicObjStatus']       = 'MusicObj does not exist: %s' % picklePath

    no_data = False
    if hasattr(dataObj,'messages'):
        if 'No data for this time period.' in dataObj.messages:
            no_data = True
            webData['good_period_warn']       = 'No data for time period. (%s)' % picklePath

    if webData['musicObjStatusClass'] == 'statusNormal' and not no_data:
        dataObj     = pydarn.proc.music.checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)
        dataSets    = dataObj.get_data_sets()
        lst = []
        for dataSet in dataSets:
            currentData = getattr(dataObj,dataSet)
            ds = {}
            ds['name']    = dataSet
            
            histList = []
            keys = currentData.history.keys()
            keys.sort()
            for key in keys:
                histList.append({'name':key,'value':currentData.history[key]})
            ds['history'] = histList

            metaList = []
            keys = currentData.metadata.keys()
            keys.sort()
            for key in keys:
                metaList.append({'name':key,'value':currentData.metadata[key]})
            ds['metadata'] = metaList

            lst.append(ds)
        webData['dataSets'] = lst

        #Send information about detected signals to the web.
        currentData = getattr(dataObj,dataSets[-1])
        if hasattr(currentData,'sigDetect'):
            sigs        = currentData.sigDetect
            webData['sigList'] = (sigs.string())

        #Stringify information about signals aready in the database...
        if rec.has_key('signals'):
            webData['sigsInDb'] = pydarn.proc.music.stringify_signal_list(rec['signals'],sort_key='serialNr')

        #Tell web if marked as a bad period.
        try:
            if not dataObj.DS000_originalFit.metadata['good_period']:
                webData['good_period_warn'] = 'WARNING: Data marked as bad period!!'
        except:
            pass


    webData['rtiplot_sDatetime'], webData['rtiplot_fDatetime']  = msc.get_default_rti_times(musicParams,dataObj)
    webData['rtiplot_yrange0'], webData['rtiplot_yrange1']      = msc.get_default_gate_range(musicParams,dataObj)
    rti_beam_list = msc.get_default_beams(musicParams,dataObj)

    rti_beam_list_str = [str(rtibm) for rtibm in rti_beam_list]
    webData['rtiplot_beams'] = ','.join(rti_beam_list_str)

    if webData['rtiplot_yrange0'] is None: webData['rtiplot_yrange0'] = 'None'
    if webData['rtiplot_yrange0'] is None: webData['rtiplot_yrange1'] = 'None'

    #See if RTI Plot exists... if so, show it!
    rtiPath     = os.path.join(musicPath,'000_originalFit_RTI.png')

    try:
        with open(rtiPath):
            webData['rtiPath']  = rtiPath
    except:
        pass

    #If kArr.png exists, show it on top!
    karrPath    = glob.glob(os.path.join(musicPath,'*karr.png'))
    if len(karrPath) > 0:
        webData['karrPath'] = karrPath[0]
    else:
        pass

    #Show all other plots...
    plots = glob.glob(os.path.join(musicPath,'*.png'))
    if plots == []:
        plotDictList = False 
    else:
        plots.sort()
        plotDictList = []
        for plot in plots:
            plotDict = {}
            plotDict['path'] = plot
            plotDict['basename'] = os.path.basename(plot)
            plotDictList.append(plotDict)
        
    webData['plots']        = plotDictList
    webData['radar']        = radar
    webData['sDate']        = sDate
    webData['fDate']        = fDate
    webData['mstid_list']   = mstid_list
    webData['homeURL']      = '/music'

    #Send the categ_manu information in webData since that info may not be included in the record info.
    webData['categ_manu_mstid'] = ''
    webData['categ_manu_quiet'] = ''
    webData['categ_manu_none']  = ''
    webData['categ_manu']       = 'Null'

    if rec.has_key('category_manu'):
        webData['categ_manu'] = rec['category_manu']
        if rec['category_manu'] == 'mstid': webData['categ_manu_mstid']  = 'checked'
        if rec['category_manu'] == 'quiet': webData['categ_manu_quiet']  = 'checked'
        if rec['category_manu'] == 'None':  webData['categ_manu_none']   = 'checked'

    #Computer prev/next urls
    nav_mode = get_nav_mode()
    if nav_mode == 'list':
        webData['nav_mode_list'] = 'checked'
    else:
        webData['nav_mode_list'] = ''

    if nav_mode == 'category':
        webData['nav_mode_categ'] = 'checked'
    else:
        webData['nav_mode_categ'] = ''

    urls = msc.get_prev_next(mstid_list,_id,mode=nav_mode)
    webData['prev_url'] = urls[0]
    webData['next_url'] = urls[1]

    webData['event_dir_url'] = 'http://sd-work1.ece.vt.edu/data/mstid/statistics/webserver/'+musicPath
    webData['source_selector']      = msc.sourcesDropDown()
    enabled_sources = msc.get_enabled_sources()
#    import ipdb; ipdb.set_trace()
    timestamp=datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    return render_template('music_edit.html'
            ,webData=webData
            ,params=params
            ,timestamp=timestamp
            ,record_list=record_list
            ,record=rec
            ,musicParams=musicParams
            ,musicParams_list=musicParams_list)

@app.route('/create_music_obj', methods=['GET'])
def create_music_obj():
    radar                   = request.args.get('radar',None,type=str)
    sTime                   = request.args.get('sTime',None,type=str)
    eTime                   = request.args.get('eTime',None,type=str)
    beamLimits_0            = request.args.get('beamLimits_0',None,type=str)
    beamLimits_1            = request.args.get('beamLimits_1',None,type=str)
    gateLimits_0            = request.args.get('gateLimits_0',None,type=str)
    gateLimits_1            = request.args.get('gateLimits_1',None,type=str)
    interpolationResolution = request.args.get('interpolationResolution',None,type=str)
    filterNumtaps           = request.args.get('filterNumtaps',None,type=str)
    firFilterLimits_0       = request.args.get('firFilterLimits_0',None,type=str)
    firFilterLimits_1       = request.args.get('firFilterLimits_1',None,type=str)
    window_data             = request.args.get('window_data',None,type=str)
    kx_max                  = request.args.get('kx_max',None,type=str)
    ky_max                  = request.args.get('ky_max',None,type=str)
    autodetect_threshold_str = request.args.get('autodetect_threshold',None,type=str)
    neighborhood_0          = request.args.get('neighborhood_0',None,type=str)
    neighborhood_1          = request.args.get('neighborhood_1',None,type=str)

    #Convert string type before sending to music object creation.
    sDatetime = datetime.datetime.strptime(sTime,'%Y-%m-%d %H:%M:%S')
    fDatetime = datetime.datetime.strptime(eTime,'%Y-%m-%d %H:%M:%S')

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
    musicPath   = msc.get_output_path(radar, sDatetime, fDatetime)
    try:
        shutil.rmtree(musicPath)
    except:
        pass

    if window_data == 'true':
        window_data = True
    else:
        window_data = False

    dataObj = msc.createMusicObj(radar.lower(), sDatetime, fDatetime
        ,beamLimits                 = beamLimits
        ,gateLimits                 = gateLimits
        ,interpolationResolution    = interpRes
        ,filterNumtaps              = numtaps 
        ,fitfilter                  = True
        )

    picklePath  = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False)


    # Create a run file. ###########################################################
    runParams = {}
    runParams['radar']              = radar.lower()
    runParams['sDatetime']          = sDatetime
    runParams['fDatetime']          = fDatetime
    runParams['beamLimits']         = beamLimits
    runParams['gateLimits']         = gateLimits
    runParams['interpRes']          = interpRes
    runParams['filter_numtaps']     = numtaps
    runParams['filter_cutoff_low']  = cutoff_low
    runParams['filter_cutoff_high'] = cutoff_high
    runParams['path']               = musicPath
    runParams['musicObj_path']      = picklePath
    runParams['window_data']        = window_data
    runParams['kx_max']             = kx_max
    runParams['ky_max']             = ky_max
    runParams['autodetect_threshold'] = autodetect_threshold
    runParams['neighborhood']        = neighborhood

    msc.Runfile(radar.lower(), sDatetime, fDatetime, runParams)

    # Generate general RTI plot for original data. #################################
    #Mark where sampling window starts and stops.
    dataObj.DS000_originalFit.metadata['timeLimits'] = [runParams['sDatetime'],runParams['fDatetime']]

    rti_beams   = msc.get_default_beams(runParams,dataObj)
    rtiPath     = os.path.join(runParams['path'],'000_originalFit_RTI.png')
    msc.plot_music_rti(dataObj,fileName=rtiPath,dataSet="originalFit",beam=rti_beams)

    result=0
    return jsonify(result=result)

@app.route('/run_music', methods=['GET'])
def run_music():
    runfile_path    = request.args.get('runfile_path',None,type=str)
    msc.run_music(runfile_path)
    msc.music_plot_all(runfile_path)

    result=0
    return jsonify(result=result)

@app.route('/music_plot_all', methods=['GET'])
def music_plot_all():
    runfile_path    = request.args.get('runfile_path',None,type=str)
    msc.music_plot_all(runfile_path)

    result=0
    return jsonify(result=result)

@app.route('/music_plot_fan', methods=['GET'])
def music_plot_fan():
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    time            = request.args.get('time',None,type=str)
    sDatetime       = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
    fanScale_0            = request.args.get('fanScale_0',None,type=str)
    fanScale_1            = request.args.get('fanScale_1',None,type=str)

    try:
        fanScale = (float(fanScale_0), float(fanScale_1))
    except:
        fanScale = None

    fileName        = '001_beamInterp_fan.png'
    msc.music_plot_fan(runfile_path,time=sDatetime,fileName=fileName,scale=fanScale)

    result=0
    return jsonify(result=result)

@app.route('/music_plot_rti', methods=['GET'])
def music_plot_rti():
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    beam            = request.args.get('beam',None,type=str)
    sTime           = request.args.get('sTime',None,type=str)
    eTime           = request.args.get('eTime',None,type=str)
    sDatetime       = datetime.datetime.strptime(sTime,'%Y-%m-%d %H:%M:%S')
    eDatetime       = datetime.datetime.strptime(eTime,'%Y-%m-%d %H:%M:%S')
    rtiScale_0      = request.args.get('rtiScale_0',None,type=str)
    rtiScale_1      = request.args.get('rtiScale_1',None,type=str)
    rtiYrange_0     = request.args.get('rtiYrange_0',None,type=str)
    rtiYrange_1     = request.args.get('rtiYrange_1',None,type=str)

    beam            = beam.split(',')
    beam            = [int(xx) for xx in beam]

    try:
        rtiScale = (float(rtiScale_0), float(rtiScale_1))
    except:
        rtiScale = None

    try:
        rtiYrange = (float(rtiYrange_0), float(rtiYrange_1))
    except:
        rtiYrange = None

    runFile         = msc.load_runfile_path(runfile_path)
    musicParams     = runFile.runParams
    musicObj_path   = musicParams['musicObj_path']
    dataObj         = pickle.load(open(musicObj_path,'rb'))

    xlim = (sDatetime,eDatetime)

    #Mark where sampling window starts and stops.
    dataObj.DS000_originalFit.metadata['timeLimits'] = [musicParams['sDatetime'],musicParams['fDatetime']]

    rtiPath     = os.path.join(musicParams['path'],'000_originalFit_RTI.png')
    msc.plot_music_rti(dataObj,fileName=rtiPath,dataSet="originalFit",beam=beam,xlim=xlim,ylim=rtiYrange,scale=rtiScale)

    result=0
    return jsonify(result=result)

@app.route('/add_music_params_db', methods=['GET'])
def add_music_params_db():
    #Get data from the webpage.
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    signals         = request.args.get('signals',None,type=str)

    #Parse list of signals.
    signal_order_list = [int(x) for x in signals.split(',')]
    signal_order_list.sort()

    #Load the runfile and the associated musicObj.
    runfile         = msc.load_runfile_path(runfile_path)
    picklePath      = runfile.runParams['musicObj_path']

    dataObj     = pickle.load(open(picklePath,'rb'))
    dataSets    = dataObj.get_data_sets()
    currentData = getattr(dataObj,dataSets[-1])

    #Pull up database record to see if there are already items stored.
    _id = ObjectId(str_id)
    event   = db[mstid_list].find_one({'_id':_id})

    if event.has_key('signals'):
        sigList     = event['signals']
        try:
            serialNr    = max([x['serialNr'] for x in sigList]) + 1
        except:
            serialNr    = 0
    else:
        sigList = []
        serialNr    = 0

#    serialNr = 0
#    sigList = []
    if hasattr(currentData,'sigDetect'):
        sigs    = currentData.sigDetect
        for order in signal_order_list:
            for sig in sigs.info:
                if sig['order'] == order:
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

    status = db[mstid_list].update({'_id':_id},{'$set': {'signals':sigList}})

    result=0
    return jsonify(result=result)

@app.route('/del_music_params_db', methods=['GET'])
def del_music_params_db():
    #Get data from the webpage.
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    signals         = request.args.get('signals',None,type=str)

    #Parse list of signals.
    signal_serialNr_list = [int(x) for x in signals.split(',')]

    #Pull up database record to see if there are already items stored.
    _id     = ObjectId(str_id)
    event   = db[mstid_list].find_one({'_id':_id})
    sigList = event['signals']

    for sig in list(sigList):
        if sig['serialNr'] in signal_serialNr_list:
            sigList.remove(sig)

    status  = db[mstid_list].update({'_id':_id},{'$set': {'signals':sigList}})

    result=0
    return jsonify(result=result)


@app.route('/update_music_analysis_status', methods=['GET'])
def update_music_analysis_status():
    #Get data from the webpage.
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    analysis_status = request.args.get('analysis_status',None,type=int)

    #Pull up database record to see if there are already items stored.
    _id     = ObjectId(str_id)
    event   = db[mstid_list].find_one({'_id':_id})

    status  = db[mstid_list].update({'_id':_id},{'$set': {'music_analysis_status':bool(analysis_status)}})
    result=0
    return jsonify(result=result)

@app.route('/add_to_detected', methods=['GET'])
def add_to_detected():
    #Get data from the webpage.
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    new_kx          = request.args.get('new_kx',None,type=float)
    new_ky          = request.args.get('new_ky',None,type=float)

    if new_kx == None: return jsonfiy(result=0)
    if new_ky == None: return jsonfiy(result=0)

    #Load the runfile and the associated musicObj.
    runfile         = msc.load_runfile_path(runfile_path)
    picklePath      = runfile.runParams['musicObj_path']
    musicPath       = runfile.runParams['path']

    dataObj     = pickle.load(open(picklePath,'rb'))
    pydarn.proc.music.add_signal(new_kx,new_ky,dataObj,dataSet='active')
    pickle.dump(dataObj,open(picklePath,'wb'))

    karrPath    = glob.glob(os.path.join(musicPath,'*karr.png'))[0]
    msc.music_plot_karr(runfile_path,karrPath)

    result=0
    return jsonify(result=result)

@app.route('/del_from_detected', methods=['GET'])
def del_from_detected():
    #Get data from the webpage.
    runfile_path    = request.args.get('runfile_path',None,type=str)
    mstid_list      = request.args.get('mstid_list',None,type=str)
    str_id          = request.args.get('_id',None,type=str)
    signals         = request.args.get('signals',None,type=str)

    #Parse list of signals.
    signal_order_list = [int(x) for x in signals.split(',')]
    signal_order_list.sort()

    #Load the runfile and the associated musicObj.
    runfile         = msc.load_runfile_path(runfile_path)
    picklePath      = runfile.runParams['musicObj_path']
    musicPath       = runfile.runParams['path']

    dataObj     = pickle.load(open(picklePath,'rb'))
    pydarn.proc.music.del_signal(signal_order_list,dataObj,dataSet='active')
    pickle.dump(dataObj,open(picklePath,'wb'))

    karrPath    = glob.glob(os.path.join(musicPath,'*karr.png'))[0]
    msc.music_plot_karr(runfile_path,karrPath)

    result=0
    return jsonify(result=result)

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=5000)
