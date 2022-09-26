#!/usr/bin/env python
# A version of the MSTID calendar plot with the polar vortex in a movie-strip form.
# Hopefully this will be a primary figure in MSTID paper 2.

import sys
import os
import glob
import datetime
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import mstid
from mstid import run_helper
import pybeamer
from mstid import polar_met, prepare_output_dirs

def gen_cal_plot_list(geo_pot_src,**kwargs):
    plot_list = []
    for raw_dct,resid_dct in zip(geo_pot_src['raw'],geo_pot_src['residuals']):
        tmp     = {}

        geo_pot = {}
        geo_pot['raw']          = raw_dct
        geo_pot['residuals']    = resid_dct
        geo_pot['mean']         = geo_pot_src['mean'][0]
        geo_pot['dt']           = raw_dct['dt']
        tmp['geo_pot']          = geo_pot

        tmp.update(kwargs)

        plot_list.append(tmp)
    return plot_list

def calendar_plot_with_polar_data(cal_plot_dct):
    png_path    = mstid.calendar_plot_with_polar_data(**cal_plot_dct)
    return png_path

# User-Defined Run Parameters Go Here. #########################################
correlate           = False
multiproc           = False
plot_amplitude      = True
plot_direction      = False

#val_key             = 'mstid_azm_dev'
val_key             = 'music_azm'
highlight_ew        = True

output_dir          = os.path.join('output','driver_timeseries')
prepare_output_dirs({0:output_dir},clear_output_dirs=True)

db_name             = 'mstid'
tunnel,mongo_port   = mstid.createTunnel()

# For MSTID amplitude plotting.
all_years           = mstid.run_helper.create_default_radar_groups_all_years()

# For MSTID Direction Plotting.
mstid_list_format   = 'music_guc_{radar}_{sDate}_{eDate}'
music_groups        = mstid.run_helper.create_default_radar_groups_all_years(mstid_format=mstid_list_format)

bmrd    = {}
bmrd['output_dir']      = os.path.join(output_dir,'beamer')
bmrd['full_title']      = 'SuperDARN MSTID Occurence and Geomagnetic Activity'
bmrd['running_title']   = 'SuperDARN MSTIDs'
beamer                  = pybeamer.Beamer(**bmrd)

mstid_reduced_inx = mstid.calculate_reduced_mstid_index(all_years,
        reduction_type='mean',daily_vals=True,
        db_name=db_name,mongo_port=mongo_port)

drivers = []

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx'] 
#drivers.append(tmp)

tmp = {}
tmp['driver']   = ['mstid_reduced_inx','smoothed_ae','omni_symh']
drivers.append(tmp)

tmp = {}
tmp['driver']   = ['mstid_reduced_inx','neg_mbar_diff']
drivers.append(tmp)

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','smoothed_ae','omni_symh','neg_mbar_diff']
#drivers.append(tmp)

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','neg_mbar_diff']
#tmp['plot_geopot_maps'] = True
#drivers.append(tmp)

for driver_dct in drivers:
    driver                  = driver_dct.get('driver','mstid_reduced_inx')
    plot_geopot_maps        = driver_dct.get('plot_geopot_maps',False)
    classification_colors   = driver_dct.get('classification_colors',True)
    section_title           = str('; '.join(driver)).upper()

    beamer.add_section(section_title)
    file_suffix = '_'.join(driver)

    for radar_groups,music_group in zip(all_years,music_groups):
        date_str            = run_helper.get_seDates_from_groups(radar_groups,date_fmt='%Y%m%d')
        season              = run_helper.get_seDates_from_groups(radar_groups,date_fmt='%Y')
        dates               = run_helper.get_seDates_from_groups(radar_groups,date_fmt=None)
        key_dates           = run_helper.get_key_dates(*dates)
        if season != '2012_2013': continue
        print 'Calendar plot: {}; Driver: {!s}'.format(date_str,driver)

        subsec_datestr  = run_helper.get_seDates_from_groups(radar_groups,date_fmt='%d %b %Y',sep=' - ')
        slide_str   = '{} / {}'.format(subsec_datestr,section_title)
        subsec_str  = subsec_datestr

        mbar_levels = [10,1]
        mbar_dict       = {}
        for mbar_level in mbar_levels:
            tmp                     = {}
            tmp['alpha']            = 0.5
            tmp['contour']          = True
            tmp['cbar_label']       = '{!s} mbar Level'.format(mbar_level)
            mbar_dict[mbar_level]   = tmp

        mbar_dict[1]['cmap']    = mstid.general_lib.truncate_colormap(matplotlib.cm.Blues,0.20)
        mbar_dict[10]['cmap']   = mstid.general_lib.truncate_colormap(matplotlib.cm.Reds,0.20)

        # The next 2 lines definatively set the scales.  Comment them out for autoscaling.
        mbar_dict[1]['scale']   = (4.3e5,4.8e5)
        mbar_dict[10]['scale']  = (2.8e5,3.1e5)

        grib_data   = []
        for mbar_level in mbar_levels:
            # The scale for each mbar and each year is set in here.
            tmp = polar_met.get_processed_grib_data(season=season,mbar_level=mbar_level)
            grib_data.append(tmp)

        if plot_amplitude:
            # Amplitude
            tmp = {}
            tmp['group_dict']           = radar_groups
            tmp['driver']               = driver
            tmp['db_name']              = db_name
            tmp['mongo_port']           = mongo_port
            tmp['output_dir']           = output_dir
            tmp['mstid_reduced_inx']    = mstid_reduced_inx 
            tmp['correlate']            = correlate
            tmp['classification_colors']= classification_colors
            cal_plot_opts               = tmp

            plot_list   = polar_met.delta_cal_plot_list(grib_data,mbar_dict=mbar_dict,
                    **cal_plot_opts)

            slide_list  = []
            png_path    = mstid.calendar_plot_lib.drivers_only(plot_list,frame_times=key_dates,
                                save_pdf=True,paper_legend=True,plot_letters=True,file_suffix='_'+file_suffix,plot_geopot_maps=plot_geopot_maps)
            print png_path
            slide_list.append(png_path)

            beamer.add_subsection(subsec_str)
            beamer.add_figs_slide(slide_str,slide_list,width=9.25)

beamer.write_latex()
beamer.make()
print "I'm done!"
