import os
import shutil

import collections

import numpy as np
import matplotlib

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=None,name=None):
    if n is None:
        n = cmap.N

    if name is None:
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval)
    
    cmap_list   = cmap(np.linspace(minval, maxval, n))
    lsc         = matplotlib.colors.LinearSegmentedColormap
    new_cmap    = lsc.from_list(name,cmap_list)
    
    return new_cmap

def combine_cmaps(cmaps,minval=0.0,maxval=1.0,n=None,name='Combined CMAP'):
    lsc             = matplotlib.colors.LinearSegmentedColormap
    new_list        = np.array([])
    new_list.shape  = (0,4)

    for cmap in get_iterable(cmaps):
        if n is None:
            N = cmap.N
        else:
            N = n

        cmap_list   = cmap(np.linspace(minval, maxval, N))
        new_list    = np.append(new_list,cmap_list,axis=0)
    
    new_cmap    = lsc.from_list(name,new_list)
    return new_cmap

def get_custom_cmap(name='blue_red'):
    if name == 'blue_red':
        l_cut       = 0.30
        u_cut       = 0.0
        blues       = truncate_colormap(matplotlib.cm.Blues_r,minval=u_cut,maxval=1.-l_cut)
        reds        = truncate_colormap(matplotlib.cm.Reds,minval=l_cut,maxval=1.-u_cut)
        combine     = [blues,reds]
        new_cmap    = combine_cmaps(combine,name=name)

    return new_cmap

def get_iterable(x):
    if isinstance(x, collections.Iterable) and not isinstance(x,basestring):
        return x
    else:
        return [x]

def generate_radar_dict():
    rad_list = []
    rad_list.append(('bks', 39.6, -81.1))
    rad_list.append(('wal', 41.8, -72.2))
    rad_list.append(('fhe', 42.5, -95.0))
    rad_list.append(('fhw', 43.3, -102.7))
    rad_list.append(('cve', 46.4, -114.6))
    rad_list.append(('cvw', 47.9, -123.4))
    rad_list.append(('gbr', 58.4, -59.9))
    rad_list.append(('kap', 55.5, -85.0))
    rad_list.append(('sas', 56.1, -103.8))
    rad_list.append(('pgr', 58.0, -123.5))

    radar_dict = {}
    for radar,lat,lon in rad_list:
        tmp                 = {}
        tmp['lat']          = lat
        tmp['lon']          = lon
        radar_dict[radar]   = tmp

    return radar_dict

def prepare_output_dirs(output_dirs={0:'output'},clear_output_dirs=False,img_extra=''):
    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' width=\'100%\'> ";')
    txt.append('}')
    txt.append('?>')
    show_all_txt_100 = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    for value in output_dirs.itervalues():
        if clear_output_dirs:
            try:
                shutil.rmtree(value)
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass
        with open(os.path.join(value,'0000-show_all_100.php'),'w') as file_obj:
            file_obj.write(show_all_txt_100)
        with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
            file_obj.write(show_all_txt)
