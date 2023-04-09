#!/usr/bin/env python
import inspect
import logging
import logging.config
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
logging.filename    = curr_file+'.log'
logging.config.fileConfig("../config/logging.conf")
log = logging.getLogger("root")

import pymongo
mongo   = pymongo.MongoClient()
db      = mongo.mstid

mstid_list  = 'paper2_wal_2012'

crsr    = db[mstid_list].find()
for item in crsr:
    blob_keys = []
    for key in item.keys():
        if 'blob_sl' in key:
            blob_keys.append(key)

    if len(blob_keys) > 0:
        for blob_key in blob_keys:
            db[mstid_list].update({},{'$unset':{blob_key:''}},multi=True)
            log.debug('Removing %s' % blob_key)
