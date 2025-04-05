#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates as md
import datetime
import music_support as msc

import os
import h5py
from hdf5_api import saveDictToHDF5, extractDataFromHDF5
from wx_pretrack import *
from auto_range_music import run_music
# import operator

# Connect to database. #########################################################
mstid_list  = 'paper2_wal_2012'
#mstid_list  = 'paper2_bks_2010_2011_0817mlt'

import pymongo
mongo       = pymongo.MongoClient()
db          = mongo.mstid
db_blob     = mongo.blob
crsr        = db[mstid_list].find().sort('sDatetime',1)

################################################################################
show_all_txt = ''' 
<?php
foreach (glob("*.png") as $filename) {
echo "<img src='$filename'><br />";
}
?>
'''
make_plots          = True #Recompute must also be True.
clear_output_dir    = True 
recompute           = True
update_blob_db      = True
output_dir          = os.path.join('output',mstid_list)
output_dirs         = {}
output_dirs['blob_line_plots']  = os.path.join(output_dir,'blob_line_plots')
output_dirs['blob_track']       = os.path.join(output_dir,'blob_track')
output_dirs['blob_dict']        = os.path.join(output_dir,'blob_dict')

for key in output_dirs:
    if clear_output_dir and os.path.exists(output_dirs[key]):
        shutil.rmtree(output_dirs[key])

    if not os.path.exists(output_dirs[key]): os.makedirs(output_dirs[key])
    with open(os.path.join(output_dirs[key],'0000-show_all.php'),'w') as file_obj:
        file_obj.write(show_all_txt)

error_file_path     = os.path.join(output_dir,'blob_track_error.txt')
if clear_output_dir and os.path.exists(error_file_path):
        os.remove(error_file_path)

################################################################################
for event in crsr:
    sTime       = event['sDatetime']
    eTime       = event['fDatetime']
    radar       = event['radar']
    filename    = msc.get_hdf5_name(radar,sTime,eTime,getPath=True)
    fig_name    = os.path.basename(filename)[:-2]
    pkl_name    = os.path.join(output_dirs['blob_dict'],fig_name+'.blob_dict.h5')

    try:
        t0  = datetime.datetime.now()
        if os.path.exists(pkl_name) and not recompute: 
            # Load blob diction if it exists. ############################################## 
            with h5py.File(pkl_name, 'r') as fl:
                blob_dict = extractDataFromHDF5(fl)
        elif not os.path.exists(pkl_name) and not recompute: 
            continue
        else:
            # Create blob dictionary. ###################################################### 
            if not os.path.exists(filename):
                event_list = [{'radar':radar, 'sDatetime':sTime, 'fDatetime':eTime}]
                run_music(event_list,only_plot_rti=True)

            now     = str(datetime.datetime.now())+':'
            evt_str = ' '.join([now,event['radar'],str(event['sDatetime']),str(event['fDatetime'])])+'\n'

            print('BLOB_TRACK START: ',evt_str)

            obj = MstidTracker(filename=filename)

            blob_dict   = {}
            for _id_int in np.arange(obj.labels)+1:
                _id = str(_id_int)
                blob_dict[_id]              = {}
                blob_dict[_id]['dt']        = []
                blob_dict[_id]['count']     = []
                blob_dict[_id]['raw_mean']  = []

                blob_dict[_id]['box_x0']    = []
                blob_dict[_id]['box_y0']    = []
                blob_dict[_id]['box_x1']    = []
                blob_dict[_id]['box_y1']    = []

            for key,frame in obj.frames.items():
                if frame.dt < sTime or frame.dt >= eTime: continue
                for item in frame.mrk_tracked.info:
                    blob_key    = str(item['id'])
                    if 'color' not in list(blob_dict[blob_key].keys()): 
                        blob_dict[blob_key]['color']  = item['color']
                    blob_dict[blob_key]['dt'].append(frame.dt)
                    blob_dict[blob_key]['count'].append(int(item['count']))
                    blob_dict[blob_key]['raw_mean'].append(float(item['raw_mean']))

                    blob_dict[blob_key]['box_x0'].append(int(item['box']['x0']))
                    blob_dict[blob_key]['box_y0'].append(int(item['box']['y0']))
                    blob_dict[blob_key]['box_x1'].append(int(item['box']['x1']))
                    blob_dict[blob_key]['box_y1'].append(int(item['box']['y1']))

        # Compute blob lifetimes. ######################################################  
        for _id in list(blob_dict.keys()):
            if len(blob_dict[_id]['dt']) == 0: continue
            blob_dict[_id]['lifetime'] = float((max(blob_dict[_id]['dt']) - min(blob_dict[_id]['dt'])).total_seconds() / 3600.) # Save to dictionary as decimal hours.

        lifetime = {}
        lifetime_min_hours = 0.75
        for _id in list(blob_dict.keys()):
            if 'lifetime' in blob_dict[_id]:
                lt = blob_dict[_id]['lifetime']
                if lt <= lifetime_min_hours: continue
                lifetime[_id] = lt

        with h5py.File(pkl_name, 'w') as fl:
            saveDictToHDF5(fl, blob_dict)

        t1      = datetime.datetime.now()
        t_tot   = t1 - t0
        print('Done: ',t_tot)

#        if len(lifetime.keys()) == 0:
#            print 'NO GOOD GS PATCHES: ' + evt_str
#            continue

        # Save to blob_db. ############################################################# 
        query   = {'radar':radar, 'sDatetime': sTime, 'fDatetime': eTime}
        update  = {'blob_dict':blob_dict, 'short_list':lifetime, 'short_list_min_hours':lifetime_min_hours}
        if update_blob_db:
            db_blob[mstid_list].update(query,{'$set':update},upsert=True)

        # Guess what... plotting here!! ################################################
        if make_plots and recompute:
            plot_list = []

            tmp  = {}
            tmp['param']    = 'count'
            tmp['ylabel']   = 'Pixels'
            plot_list.append(tmp)

            tmp  = {}
            tmp['param']    = 'raw_mean'
            tmp['ylabel']   = 'Mean Power [dB]'
            plot_list.append(tmp)

            tmp  = {}
            tmp['param']    = 'box_y1'
            tmp['ylabel']   = 'Upper Bound'
            plot_list.append(tmp)

            tmp  = {}
            tmp['param']    = 'box_y0'
            tmp['ylabel']   = 'Lower Bound'
            plot_list.append(tmp)

            nx_plots    = 1
            ny_plots    = len(plot_list)
            figsize     = (10*nx_plots,3*ny_plots)


            fig         = plt.figure(figsize=figsize)
            plot_nr     = 0
            for plot in plot_list:
                plot_nr += 1

                prm     = plot['param']

                axis    = fig.add_subplot(ny_plots,nx_plots,plot_nr)
                for _id in lifetime:
                    item    = blob_dict[_id]
                    axis.plot(item['dt'],item[prm],color=item['color'],label=str(_id))

                axis.set_xlim(sTime,eTime)
                axis.legend(loc='upper left')#,prop={'size':8})
                axis.set_ylabel(plot['ylabel'])
                axis.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

            fig.text(0.5,1.01,fig_name,ha='center',size=18)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dirs['blob_line_plots'],fig_name+'.png'),bbox_inches='tight')
            fig.clf()

            ################################################################################
            fig     = plt.figure(figsize=(20,6))
            nx_plot = 16
            ny_plot = 1
            plot_nr = 0

            plot_times  = [sTime + x*datetime.timedelta(minutes=15) for x in range(8)]
            for pt in plot_times:
                apt  = min(obj.currentData.time,key=lambda date : abs(pt-date))
                inx  = str(np.where(obj.currentData.time == apt)[0][0])

                plot_nr += 1
                axis    = fig.add_subplot(ny_plot,nx_plot,plot_nr)
                img     = obj.frames[inx].gry_rgb
                axis.imshow(img)

                axis.set_title(apt.strftime('%H%M UT'))
                
                if plot_nr != 1:
                    axis.get_yaxis().set_visible(False)

            for pt in plot_times:
                apt  = min(obj.currentData.time,key=lambda date : abs(pt-date))
                inx  = str(np.where(obj.currentData.time == apt)[0][0])

                plot_nr += 1
                axis    = fig.add_subplot(ny_plot,nx_plot,plot_nr)
                img     = obj.frames[inx].mrk_tracked.markers_rgb
                axis.imshow(img)

                axis.set_title(apt.strftime('%H%M UT'))
                
                if plot_nr != 1:
                    axis.get_yaxis().set_visible(False)


            fig.text(0.5,1.01,fig_name,ha='center',size=18)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dirs['blob_track'],fig_name+'.png'),bbox_inches='tight')
            fig.clf()

    except:
        now = str(datetime.datetime.now())+':'
        err =' '.join([now,event['radar'],str(event['sDatetime']),str(event['fDatetime'])])+'\n'
        print('BLOB_TRACK ERROR: '+err)
        with open(error_file_path,'a') as error_file:
            error_file.write(err)
