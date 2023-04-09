#!/usr/bin/env python
import pymongo
mongo   = pymongo.MongoClient()
db      = mongo.mstid

mstid_list  = 'paper2_wal_2012'

crsr    = db[mstid_list].find({'category_manu':'quiet'})

good        = 0
bad         = 0
the_ugly    = 0

for item in crsr:
    val = item.get('good_period')
    if val is None:
        the_ugly += 1
        print 'THE UGLY: ',item['radar'],item['sDatetime'],item['fDatetime']
    elif val == False:
        bad += 1
        print 'THE BAD: ',item['radar'],item['sDatetime'],item['fDatetime']
    else:
        good += 1
