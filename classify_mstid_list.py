#!/usr/bin/env python
import os
import sys
import datetime
import matplotlib
matplotlib.use('Agg')

import mstid
date_fmt = '%Y%m%d%H%M'
radar                   = sys.argv[1] 
list_sDate              = datetime.datetime.strptime(sys.argv[2],date_fmt)
list_eDate              = datetime.datetime.strptime(sys.argv[3],date_fmt)
mstid_list              = sys.argv[4]
sort_key                = sys.argv[5]
data_path               = sys.argv[6]
classification_path     = sys.argv[7]
db_name                 = sys.argv[8]
mongo_port              = int(sys.argv[9])

output_dir  = os.path.join(classification_path,'results',mstid_list)
cache_dir   = os.path.join(classification_path,'cache',mstid_list)

dirs        = {}
dirs[0]     = output_dir
dirs[1]     = cache_dir
mstid.prepare_output_dirs(dirs,clear_output_dirs=True)

data_dict   = mstid.classify.load_data_dict(mstid_list,data_path,cache_dir=cache_dir,test_mode=False,
        db_name=db_name,mongo_port=mongo_port)

if data_dict is None:
    fout = os.path.join(output_dir,'spectrum_messages.txt')
    with open(fout,'w') as fl:
        msg = 'No data for given time period. Spectral classification not possible.'
        print(msg)
        fl.write(msg)
        quit()

data_dict   = mstid.classify.sort_by_spectrum(data_dict,sort_key)
data_dict   = mstid.classify.classify_mstid_events(data_dict)

mstid.classify.rcss(data_dict,classification_path=classification_path)

plot_nr     = 1
filename    = '{:03d}_spectral_plot.png'.format(plot_nr)
mstid.classify.spectral_plot(data_dict,output_dir=output_dir,plot_all_spect_mean=True,filename=filename)

plot_nr     += 1
filename    = '{:03d}_spectral_plot_mean_subtracted.png'.format(plot_nr)
mstid.classify.spectral_plot(data_dict,output_dir=output_dir,filename=filename,subtract_mean=True)

plot_nr     += 1
filename    = '{:03d}_spectral_plot_mean_subtracted_ranked.png'.format(plot_nr)
mstid.classify.spectral_plot(data_dict,output_dir=output_dir,subtract_mean=True,color_key='spectral_sort',filename=filename)
