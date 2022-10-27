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

import utils

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

mongo         = pymongo.MongoClient()
db            = mongo.mstid
coll_name     = 'psd_manual'

fileList = []
fileList.append('/data/waver/lib/idl/doypsd/20111101-bks.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-cvw.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-fhe.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-gbr.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-kap.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-pgr.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-sas.sav')
fileList.append('/data/waver/lib/idl/doypsd/20111101-wal.sav')

db[coll_name].drop()
db['intpsd'].drop()

for fl in fileList:
  data        = readsav(fl)
  data['dt']  = np.array(utils.timeUtils.julToDatetime(data['julvec']))
  radar       = data['wave_dataproc_info']['radar'][0]

  nrecs = len(data['dt'])
  for kk in range(nrecs):
    forDb = {'radar': radar, 'date': data['dt'][kk], 'intpsd': float(data['psdvec'][kk])}
    print 'Inserting: ',radar,data['dt'][kk],float(data['psdvec'][kk])
    db['intpsd'].insert(forDb)
  
  currentDate = datetime.datetime(2010,11,1)
  endDate     = datetime.datetime(2011,11,1)
  while currentDate < endDate: 
    nextDate = currentDate + datetime.timedelta(hours=2)

    cursor      = db['intpsd'].find({'radar':radar, 'date': {'$gte': currentDate, '$lt':nextDate}})
    psdObj_list  = [x for x in cursor]

    oneDayList = []
    for psdObj in psdObj_list:
      oneDayList.append(psdObj['intpsd'])

    if oneDayList != []:
      intpsd_sum  = float(np.sum(oneDayList))
      intpsd_max  = float(np.max(oneDayList))
      intpsd_mean = float(np.mean(oneDayList))
    else:
      intpsd_sum  = 'NaN'
      intpsd_max  = 'NaN'
      intpsd_mean = 'NaN'
    print radar,currentDate,endDate,len(oneDayList),intpsd_sum,intpsd_max
    record= {'date':currentDate, 'radar':radar, 'intpsd_sum': intpsd_sum, 'intpsd_max': intpsd_max, 'intpsd_mean': intpsd_mean}
    db[coll_name].insert(record)

    currentDate = nextDate
