#!/usr/bin/env python
import os,sys
sys.path.append('/data/mypython')

import pickle
import glob
import datetime

import matplotlib
matplotlib.use('Agg')
import numpy as np

import handling
import multi_radar_music_support as msc
import multiprocessing

def plot_maps(input_tuple):
    multi_radar_dict    = input_tuple[0]
    time                = input_tuple[1]
    full_path           = input_tuple[2]

#    panels              = ['raw','direct']
#    panels              = ['raw','filtered']
#    panels              = ['raw','direct','interp','filtered-direct']
    panels              = ['raw','interp','filtered-direct','filtered-interp']
#    panels              = ['filtered-direct']
    msc.merge_fan_compare(multi_radar_dict,time=time,full_path=full_path,panels=panels)

def main(data_path,parallel=False):

    data_files  = glob.glob(os.path.join(data_path,'*.p'))
    data_files.sort()
    
    clear_output_dirs = True
    for data_file in data_files:
        basename    = os.path.basename(data_file)
        sTime_str   = basename[:13]
        eTime_str   = basename[14:27]
        radar_pair  = basename[28:35]

        sTime       = datetime.datetime.strptime(sTime_str,'%Y%m%d.%H%M')
        eTime       = datetime.datetime.strptime(eTime_str,'%Y%m%d.%H%M')

        with open(data_file,'rb') as tmp:
            multi_radar_dict = pickle.load(tmp)

        base_merge_type = 'direct'
        sub_dir         = 'merge_compare_movie'
        dict_key        = '-'.join([radar_pair,base_merge_type])
        output_dir      = os.path.split(multi_radar_dict[dict_key]['musicPath'])[0]
        output_dir      = output_dir.replace(base_merge_type,'compare_movie')

        handling.prepare_output_dirs({0:output_dir},clear_output_dirs=clear_output_dirs)
        if clear_output_dirs: clear_output_dirs = False

        times   = multi_radar_dict[dict_key]['dataObj'].DS004_merged_direct.time
        times.sort()

        times_arr   = np.array(times)
        times_arr   = times_arr[np.logical_and(times_arr >= sTime,times_arr < eTime)]
        times       = times_arr.tolist()

        full_paths  = []
        for nr,time in enumerate(times):
            full_path = os.path.join(output_dir,'{}.png'.format(time.strftime('%Y%m%d_%H%M')))
            full_paths.append(full_path)

        input_tuples = [(multi_radar_dict,time,full_path) for time, full_path in zip(times,full_paths)]

        if parallel:
            pool = multiprocessing.Pool()
            pool.map(plot_maps,input_tuples)
            pool.close()
            pool.join()
        else:
            for input_tuple in input_tuples:
                plot_maps(input_tuple)

if __name__ == '__main__':
    data_path   = '/data/mstid/statistics/paper2/multi_radar_music_scripts/multi_radar_music/fhw_fhe/compare_pickle'
    parallel    = False
    parallel    = True
    main(data_path,parallel=parallel)
