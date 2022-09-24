#!/usr/bin/env python
# Used for updating mongo db with event information.
# This wrapper script is used to eliminate memory leaks.

import sys
import datetime

import more_music
import mongo_tools

date_fmt    = '%Y%m%d%H%M'
radar       = sys.argv[1]
sTime       = datetime.datetime.strptime(sys.argv[2],date_fmt)
eTime       = datetime.datetime.strptime(sys.argv[3],date_fmt)
data_path   = sys.argv[4]
mstid_list  = sys.argv[5]
db_name     = sys.argv[6]
mongo_port  = int(sys.argv[7])

dataObj     = more_music.get_dataObj(radar,sTime,eTime,data_path)
status      = mongo_tools.dataObj_update_mongoDb(radar,sTime,eTime,dataObj,
        mstid_list,db_name,mongo_port)
