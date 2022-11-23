#!/usr/bin/env python
import os
import glob
import pickle

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

#Find all of the sorted logreg_rank files in a logistic regression directory.
base_dir = 'output/logreg'
logreg_dirs    = [x for x in glob.glob(os.path.join(base_dir,'*')) if os.path.isdir(x)]

#Load each of the files and store them in a ranks dictionary.
ranks       = {}
nr_combos_list  = []
for logreg_dir in logreg_dirs:
    pckl_fl     = os.path.join(logreg_dir,'rank','sorted_rank.p')
    mstid_list  = os.path.basename(logreg_dir)
    if os.path.exists(pckl_fl):
        with open(pckl_fl) as fl:
            ranks[mstid_list] = pickle.load(fl)

        nr_combos_list.append(len(ranks[mstid_list]))

if np.unique(nr_combos_list).size != 1: raise Exception('Lists don\'t have equal numbers of combinations!!!')
nr_combos = nr_combos_list[0]


#Find the topmost feature combinations for each mstid_list.
top_most_pct    = 5.
top_most        = int(top_most_pct * 0.01 * nr_combos)
top = {}
for key,val in ranks.items():
    top[key] = []
#    print key,len(val)
    for item in val[:top_most]:
        top[key].append(item['feature_list'])

#for key,val in top.iteritems():
#    print ''
#    print '=== {mstid_list} =='.format(mstid_list=key)
#    for item in val:
#        print item

#For now, we can only handle comparing 2 lists.
key_1,key_2 = list(top.keys())

#Find out which top features are shared in both mstid_lists.
#Save useful information to a shared_items dict.
shared_items    = {}
for item in top[key_1]:
    if item in top[key_2]:
        shared_items[item] = {}

for key,val in shared_items.items():
    for mstid_list in list(top.keys()):
        rnk = top[mstid_list].index(key)
        shared_items[key][mstid_list] = {}
        shared_items[key][mstid_list]['rank'] = rnk

        for item in ranks[mstid_list]:
            if item['feature_list'] == key:
                shared_items[key][mstid_list]['mmpc'] = item['mean_mean_pct_correct']
                break
            else:
                continue

#Calculate a shared rank.
shared_rank_list = []
for key,val in shared_items.items():
    val['shared_rank'] = 0
    for mstid_list in list(top.keys()):
        val['shared_rank'] += shared_items[key][mstid_list]['rank'] 

    shared_rank_list.append({'feature_list':key,'shared_rank':val['shared_rank']})

shared_rank_list = sorted(shared_rank_list,key=lambda k: k['shared_rank'])


#Print a nice report of our findings.
#Here we print the names of the two mstid_lists along with index keys.
mstid_list_inxs = list(range(len(list(top.keys()))))
mstid_lists = list(zip(mstid_list_inxs,list(top.keys())))
for inx,mstid_list in mstid_lists:
    print(inx, mstid_list)

#Now print the name of the lists, the rank and mean MPC associated with each feature set.
print('')
print('Comparing the top {top_most_pct:.1f}% rankings ({top_most:d} of {nr_combos:d})'.format(top_most_pct=float(top_most_pct),top_most=top_most,nr_combos=nr_combos))
print('\t\t'.join(['List {inx}'.format(inx=str(x)) for x in mstid_list_inxs]))
for item in shared_rank_list:
    key = item['feature_list']
    val = shared_items[key]
    txt = []
    for inx,mstid_list in mstid_lists:
        txt_1 = '{rank:d} ({mmpc:.2f}%)'.format(rank=val[mstid_list]['rank'],mmpc=val[mstid_list]['mmpc'])
        txt.append(txt_1)
    txt.append(str(key))
    print('\t'.join(txt))
