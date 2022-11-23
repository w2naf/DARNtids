#!/usr/bin/env python
#Script to inspect various things about pickled blob track dictionaries.
#Right now, it prints out the name of every file that does not have a tracked blobs
#of a minimum lifetime.

# import music_support as msc

import os
import pickle
import glob

#mstid_list  = 'paper2_wal_2012'
mstid_list  = 'paper2_bks_2010_2011_0817mlt'

output_dir          = os.path.join('output',mstid_list)
output_dirs         = {}
output_dirs['blob_dict']        = os.path.join(output_dir,'blob_dict')

files   = glob.glob(os.path.join(output_dirs['blob_dict'],'*.blob_dict.p'))

good    = 0
bad     = 0

for fl_name in files:
    with open(fl_name,'rb') as fl:
        blob_dict = pickle.load(fl)

    lifetime = {}
    for _id in list(blob_dict.keys()):
        if 'lifetime' in blob_dict[_id]:
            lt = blob_dict[_id]['lifetime']
            if lt < 0.75: continue
            lifetime[_id] = lt

    if len(list(lifetime.keys())) == 0:
        print('Bad: %s' % fl_name)
        bad += 1
    else:
        good += 1

print('Good: %d; Bad: %d' % (good,bad))
