#!/usr/bin/env python
"""
fitexfilter.py
Nathaniel Frissell - 5 February 2024

This script will apply A.J. Ribeiro's FITEX median filter to a set of
SuperDARN FITACF or FITEX files.

A binary of A.J. Ribeiro's fitexfilter is included in this directory.
"""

import os
import glob

import datetime
import numpy as np

import subprocess
import tempfile

import multiprocessing
import tqdm


def find_to_be_processed(sDate,eDate,radars=None,
    base_in     = '/data/sd-data',
    in_type     = 'fitacf',
    base_out    = '/data/sd-data_fitexfilter',
    out_type    = 'fitexfilter.fitacf'):

    years       = sDate.year + np.arange((eDate.year - sDate.year)+1)

    to_be_processed = []
    for year in tqdm.tqdm(years,desc='Years',dynamic_ncols=True):
        year_path   = os.path.join(base_in,str(year),in_type)
        radar_paths = glob.glob(os.path.join(year_path,'*'))

        for radar_path in tqdm.tqdm(radar_paths,desc='Radars',dynamic_ncols=True,leave=False):
            radar = os.path.basename(radar_path)

            # Only include specific radars if asked.
            if radars is not None:
                if radar not in radars: continue

            files_in = glob.glob(os.path.join(radar_path,'*.{!s}.bz2'.format(in_type)))

            for file_in in tqdm.tqdm(files_in,desc='Files',dynamic_ncols=True,leave=False):
                bname_in = os.path.basename(file_in)
                time_in  = datetime.datetime.strptime(bname_in[:13],'%Y%m%d.%H%M')

                if time_in < sDate or time_in > eDate:
                    continue

                bname_out   = bname_in.replace(in_type,out_type)
                dir_out     = os.path.join(base_out,str(year),in_type,radar)
                file_out    = os.path.join(dir_out,bname_out)

                # Don't process a file if it already exists.
                if os.path.exists(file_out):
                    continue

                to_be_processed.append( (file_in, file_out) )

    return to_be_processed

def generate_directories(to_be_processed):
    dirs    = []
    for tbp in to_be_processed:
        out_dir = os.path.split(tbp[1])[0]
        if out_dir not in dirs:
            dirs.append(out_dir)

    for dr in dirs:
        os.makedirs(dr,exist_ok=True)

def despeckle(tbp):
    tmp_dir     = tempfile.gettempdir()

    file_in     = tbp[0]
    bname_in    = os.path.basename(file_in)

    file_out    = tbp[1]
    bname_out   = os.path.basename(file_out)

    raw_uncmp   = os.path.join(tmp_dir,bname_in.rstrip('.bz2'))
#    dspk_uncmp  = os.path.join(tmp_dir,bname_out.rstrip('.bz2'))

    # Uncompress raw file
    cmd = 'bzcat {!s} > {!s}'.format(file_in,raw_uncmp)
    subprocess.run(cmd,shell=True)

    cmd = 'fitexfilter {!s} | bzip2 -c > {!s}'.format(raw_uncmp,file_out)
    subprocess.run(cmd,shell=True)
    
    os.remove(raw_uncmp)

if __name__ == "__main__":

    N_proc      = 60
    sDate       = datetime.datetime(2010,1,1)
    eDate       = datetime.datetime(2023,1,1)

#    radars      = ['pgr','sas','kap','gbr','cvw','cve','fhw','fhe','bks','wal']
    radars      = None # Set to None to process all radars.

    print('Finding files to filter...')
    to_be_processed = find_to_be_processed(sDate,eDate,radars=radars)
    print('Total files to filter: {!s}\n'.format(len(to_be_processed)))

    generate_directories(to_be_processed)

    print('Filtering files...')
    if N_proc == 1:
        for tbp in tqdm.tqdm(to_be_processed,dynamic_ncols=True):
            despeckle(tbp)
    elif N_proc > 1:
        with multiprocessing.Pool(N_proc) as pl:
            rr = list(tqdm.tqdm(pl.imap(despeckle,to_be_processed),total=len(to_be_processed),dynamic_ncols=True))
