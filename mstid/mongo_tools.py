import os
import datetime

import inspect
import subprocess
import multiprocessing
import socket
import shlex
import time
import sh

import numpy as np
import pandas as pd

import ephem # pip install pyephem (on Python 2)
             # pip install ephem   (on Python 3)
             # conda install -c anaconda ephem

import pymongo

import pydarn
import pyDARNmusic

from . import more_music
from .general_lib import generate_radar_dict


radar_dict  = generate_radar_dict()

class FakeTunnel(object):
    def kill(self):
        pass

def createTunnel(localport=27018,remoteport=27017,
        identityfile='~/.ssh/id_rsa',user=None,
        server='localhost'):
    """Create SSH Tunnels for Database connections"""

    if socket.gethostname() == 'localhost':
        return FakeTunnel(), remoteport

    identityfile    = os.path.expanduser(identityfile)
    user            = os.getenv('USER')

    port_fwd     = "{!s}:127.0.0.1:{!s}".format(localport,remoteport)
    sshTunnelCmd = "ssh -N -L {!s} -i {!s} {!s}@{!s}".format(
                port_fwd, identityfile, user, server
        )

    #### Close any previously open tunnels.
    # This is probably getting a little OS-dependent here...
    ps = sh.ps('-ef')
    for line in ps.stdout.splitlines():
        line_str = line.decode('utf-8')
        if port_fwd not in line_str: continue

        pid = int(line_str.split()[1])
        os.kill(pid,15)

    #### Create the new tunnel.
    args = shlex.split(sshTunnelCmd)
    tunnel = subprocess.Popen(args)

    time.sleep(2)  # Give it a couple seconds to finish setting up

    # return the tunnel so you can kill it before you stop
    # the program - else the connection will persist 
    # after the script ends
    return tunnel, localport

def open_mongo_tunnel(db_name='mstid',host='localhost',port=1247):
    key_path    = os.path.join(os.getenv('HOME'),'.ssh','id_rsa')
    rsa_key     = sshtunnel.paramiko.RSAKey.from_private_key_file(key_path)

    server  = sshtunnel.SSHTunnelForwarder(
                ssh_address     = (host,port),
                ssh_private_key = rsa_key,
                ssh_username    = os.getenv('USER'),
                remote_bind_address=('127.0.0.1', 27017))

    try:
        server.start()
        print('Connected to {}!'.format(host))
    except:
        print('Connection FAILURE with {}!'.format(host))
        return 

    return server.local_bind_port

def solartime(observer, sun=ephem.Sun()):
    # From: http://stackoverflow.com/questions/13314626/local-solar-time-function-from-utc-and-longitude
    sun.compute(observer)
    # sidereal time == ra (right ascension) is the highest point (noon)
    hour_angle = observer.sidereal_time() - sun.ra
    ephem_slt = ephem.hours(hour_angle + ephem.hours('12:00')).norm  # norm for 24h
    # Note: ephem.hours is a float number that represents an angle in radians and converts to/from a string as "hh:mm:ss.ff".
    return ephem_slt/(2.*np.pi) * 24

def generate_mongo_list_from_list(mstid_list,db_name,mongo_port,
        input_mstid_list,input_db_name,input_mongo_port,
        category=None):
    """
    Copy an existing MongoDB MSTID List into a new MSTID List, keeping only the following fields:
        'radar'
        'date'
        'sDatetime'
        'fDatetime'
        'lat'
        'lon'
        'slt'
        'mlt'
        'height_km'
        'gscat'
        'category_auto'
        'category_manu'
    """

    #### Connect to output and input databases.
    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]

    input_mongo = pymongo.MongoClient(port=input_mongo_port)
    input_db    = input_mongo[input_db_name]

    #### Keep the listTracker up-to-date for the web tool.
    count   = db.listTracker.count_documents({'name': mstid_list})
    if count == 0:
        db.listTracker.insert_one({'name': mstid_list})

    #### Clean out the output database.
    db[mstid_list].drop()

    #### Identify which fields to keep.
    keep    = []
    keep.append('radar')
    keep.append('date')
    keep.append('sDatetime')
    keep.append('fDatetime')
    keep.append('lat')
    keep.append('lon')
    keep.append('slt')
    keep.append('mlt')
    keep.append('height_km')
    keep.append('gscat')
    keep.append('category_auto')
    keep.append('category_manu')

    #### Get the possible events to add to the new list.
    crsr    = input_db[input_mstid_list].find()

    if category is not None:
        # Make category iterable.
        category = np.array(category)
        if category.shape == ():
            category.shape = (1,)
        category = category.tolist()

    # Insert new entry into db.
    for item in crsr:
        if category is not None:
            # Allow for a list of categories.
            cat_good = False
            for cat in category:
                this_cat = item.get('category_manu')
                if cat.lower() == 'unclassified':
                    if this_cat is not None: continue
                if cat.lower() == 'none':
                    if this_cat != 'none': continue
                if cat.lower() == 'mstid':
                    if this_cat != 'mstid': continue
                if cat.lower() == 'quiet':
                    if this_cat != 'quiet': continue

                cat_good = True
                break

            if not cat_good: continue

        record  = {}
        for key in keep:
            record[key] = item.get(key)

        record['intpsd_sum']    = 'NaN'
        record['intpsd_max']    = 'NaN'
        record['intpsd_mean']   = 'NaN'

        db[mstid_list].insert_one(record)

    mongo.close()

def generate_mongo_list(mstid_list,radar,list_sDate,list_eDate,
        lat=None,lon=None,slt_range=(6,18),height=350.,timedelta=datetime.timedelta(hours=2),
        db_name='mstid',mongo_port=27017,**kwargs):
    """
    Generates a MongoDB collection with one entry for every period being studied. A single
    collection is defined for a single radar and range of dates. While any name may be given
    to the collection through the mstid_list argument, a typical name would be in the form of:
        guc_bks_20121101_20130501

    Where 'guc' stands for Grand Unified Classification, 'bks' is the radar code, '20121101' is
    the start date of the collection, and '201305001' is the end date of the collection.

    This function generates one MongoDB document in the collection for every event period to be
    studied. The document are created with the following fields:
            'date':             currentDate
            'sDatetime':        currentDate     # Beginning of event period
            'fDatetime':        nextDate        # End of event period
            'radar':            radar           # Radar code
            'intpsd_sum':       'NaN'           # Placeholder for integrated Power Spectral Density summation
            'intpsd_max':       'NaN"           # Placeholder for integrated Power Spectral Density maximum
            'intpsd_mean':      'NaN'           # Placeholder for integrated Power Spectral Density mean
            'lat':              lat             # Center of data latitude
            'lon':              lon             # Center of data longitude
            'slt':              slt             # Solar Local Time of lat/lon calcuated by this routine using pyEphem
            'mlt':              mlt             # Magnetic Local Time of lat/lon calcuated by this routine using aacgmv2
            'gscat':            1               # Ground Scatter Flag set to 1 (TODO: Take in gscat as arguement rather than force to 1)
            'category_auto':    'None'          # Placeholder for MSTID classification
            'height_km':        height          # Height in kilometers used to calculate SLT and MLT.

    WARNING: This function will delete and overwrite the MongoDB collection specified in mstid_list.

    Arguments:
        mstid_list: <str> Name of MongoDB collection to be created. Existing collection will
            be deleted before new collection is created.
        radar:      <str> SupderDARN radar three-letter code.
        list_sDate: <datetime.datetime> Start datetime of list
        list_eDate: <datetime.datetime> End datetime of list
        lat:        <float> Latitude of center of radar data Field of View
        lon:        <float> Longitude of center of radar data Field of View
        slt_range:  <(6,18)> Range of Solar Local Times to keep. If an event is not within the
            Solar Local Time range specified, it is not added to the list. SLT of an event is
            calculated using pyEphem based on the specified lat, lon, height, and start time
            of the event.
        height:     <350.> Height of observation in kilometers.
        timedelta:  <datetime.timedelta(hours=2)> Duration of event.
        db_name:    <'mstid'> Name of MongoDB database to use.
        mongo_port: <27017> Port number to connect to MongoDB server.
        **kwargs:   No kwargs used by this function. This is here to ignore any other keyword
            arguements passed to the function.
    """

    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]
    count   = db.listTracker.count_documents({'name': mstid_list})

    if count == 0:
        db.listTracker.insert_one({'name': mstid_list})

    # WARNING!  Double check the next line before running this script! #############
    db[mstid_list].drop()

    if lat is None: lat = radar_dict[radar]['lat']
    if lon is None: lon = radar_dict[radar]['lon']
    
    o           = ephem.Observer()
    o.lon       = np.radians(lon)
    o.lat       = np.radians(lat)
    o.elevation = height*1000. # pyEphem expects height in m.
    
    currentDate = list_sDate
    while currentDate < list_eDate: 
        nextDate = currentDate + timedelta

        tm              = currentDate
        mlat, mlon, r   = pydarn.utils.coordinates.aacgmv2.convert_latlon(lat,lon,height,tm,'G2A')
        mlt             = (pydarn.utils.coordinates.aacgmv2.convert_mlt(mlon,tm))[0]

        o.date      = tm
        slt         = solartime(o)

        if slt_range is not None:
            if slt < slt_range[0] or slt >= slt_range[1]:
                print(radar,currentDate,nextDate,slt,': OUT OF SLT RANGE!')
                currentDate = nextDate
                continue
            else:
                print(radar,currentDate,nextDate,slt)

        intpsd_sum  = 'NaN'
        intpsd_max  = 'NaN'
        intpsd_mean = 'NaN'

        crsr = db[mstid_list].find_one({'date':currentDate, 'sDatetime': currentDate, 'fDatetime': nextDate, 'radar':radar})
        if not crsr:
            record= {'date':currentDate, 'sDatetime': currentDate, 'fDatetime': nextDate, 'radar':radar,
                     'intpsd_sum': intpsd_sum, 'intpsd_max': intpsd_max, 'intpsd_mean': intpsd_mean,
                     'lat': lat, 'lon': lon, 'slt': slt, 'mlt': mlt,'gscat': 1, 'category_auto':'None',
                     'height_km': height}
            db[mstid_list].insert_one(record)

        currentDate = nextDate
    mongo.close()

def dataObj_update_mongoDb(radar,sTime,eTime,dataObj,
        mstid_list,db_name='mstid',mongo_port=27017,**kwargs):
    if mstid_list is None:
        return

    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]

    srch_dct    = {'radar':radar,'sDatetime':sTime,'fDatetime':eTime}
    item        = db[mstid_list].find_one(srch_dct)
    
    # Add record if not present.
    if item is None:
        db[mstid_list].insert_one(srch_dct)
        item    = db[mstid_list].find_one(srch_dct)

    _id = item['_id']

    # Delete certain loaded information so we can start fresh.
    delete_list  = ['no_data','good_period','signals','total_spec','dom_spec','prm']
    delete_list += ['orig_rti_cnt', 'orig_rti_possible', 'orig_rti_fraction', 
                    'orig_rti_mean', 'orig_rti_median', 'orig_rti_std']

    for del_key in delete_list:
        if del_key in item:
            status = db[mstid_list].update_one({'_id':_id},{'$unset': {del_key: 1}})

    # Set missing data flag. #######################################################
    if dataObj is None:
        good_period = False
    else:
        good_period = True

    if hasattr(dataObj,'messages'):
        if 'No data for this time period.' in dataObj.messages:
            status      = db[mstid_list].update_one({'_id':_id},{'$set': {'no_data':True} })
            good_period = False

    # Check the data quailty with basic check. #####################################
    if good_period:
        good_period = dataObj.DS000_originalFit.metadata.get('good_period')

    status  = db[mstid_list].update_one({'_id':_id},{'$set': {'good_period': bool(good_period)} })
    if not good_period:
        mongo.close()
        return

    # Store summary RTI info into db. ##############################################
    dct = more_music.get_orig_rti_info(dataObj,sTime,eTime)

    # Update Terminator Fraction Information ####################################### 
    currentData         = pyDARNmusic.getDataSet(dataObj,'active')
    if hasattr(currentData,'terminator'):
        bools               = np.logical_and(currentData.time > sTime,
                                             currentData.time < eTime)
        real_terminator     = currentData.terminator[bools,:,:] #Account for the fact that the active data array may have been zeropadded. 
        terminator_fraction = np.sum(real_terminator)/float(real_terminator.size)

        tmp = {'terminator_fraction':terminator_fraction}
        dct.update(tmp)

    # Update Sig List ############################################################## 
    currentData         = pyDARNmusic.getDataSet(dataObj,'active')
    if hasattr(currentData,'sigDetect'):
        sigs    = currentData.sigDetect
        sigs.reorder()

        sigList     = []
        serialNr    = 0
        for sig in sigs.info:
            sigInfo = {}
            sigInfo['order']    = int(sig['order'])
            sigInfo['kx']       = float(sig['kx'])
            sigInfo['ky']       = float(sig['ky'])
            sigInfo['k']        = float(sig['k'])
            sigInfo['lambda']   = float(sig['lambda'])
            sigInfo['azm']      = float(sig['azm'])
            sigInfo['freq']     = float(sig['freq'])
            sigInfo['period']   = float(sig['period'])
            sigInfo['vel']      = float(sig['vel'])
            sigInfo['max']      = float(sig['max'])
            sigInfo['area']     = float(sig['area'])
            sigInfo['serialNr'] = serialNr
            sigList.append(sigInfo)
            serialNr = serialNr + 1

        tmp = {'signals':sigList}
        dct.update(tmp)

    status  = db[mstid_list].update_one({'_id':_id},{'$set': dct})
    mongo.close()
    return status

def updateDb_mstid_list_event(event_tuple):
    path   = os.path.split(inspect.getfile(inspect.currentframe()))[0]

    radar, sTime, eTime, data_path, mstid_list, db_name, mongo_port = event_tuple
    print('Updating DB: {} {} {!s} {!s} {} {!s}'.format(
            mstid_list,radar,sTime,eTime,db_name,mongo_port
            ))

    date_fmt    = '%Y%m%d%H%M'
#    cmd        = []
#    cmd.append(os.path.join(path,'mstid_list_updateDb.py'))
#    cmd.append(radar)
#    cmd.append(sTime.strftime(date_fmt))
#    cmd.append(eTime.strftime(date_fmt))
#    cmd.append(data_path)
#    cmd.append(mstid_list)
#    cmd.append(db_name)
#    cmd.append(str(mongo_port))
#
#    print(' '.join(cmd))
#    subprocess.check_call(cmd)
    
    dataObj     = more_music.get_dataObj(radar,sTime,eTime,data_path)
    if dataObj is None:
        print("No valid dataObj for {} {} - skipping update.".format(radar, sTime))
        return
    status      = dataObj_update_mongoDb(radar,sTime,eTime,dataObj,mstid_list,
                    db_name,mongo_port)

def updateDb_mstid_list(mstid_list,
        db_name='mstid',mongo_port=27017,data_path='music_data/music',
        multiproc=True,nprocs=None,**kwargs):

    print('updateDb_mstid_list')
    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]
    crsr    = db[mstid_list].find()

    event_list  = []
    for item in crsr:
        radar       = item.get('radar')
        sTime       = item.get('sDatetime')
        eTime       = item.get('fDatetime')

        tmp = (radar, sTime, eTime, data_path, mstid_list, db_name, mongo_port)
        event_list.append(tmp)

    if multiproc:
        pool = multiprocessing.Pool(nprocs)
        pool.map(updateDb_mstid_list_event,event_list)
        pool.close()
        pool.join()
    else:
        for event in event_list:
            updateDb_mstid_list_event(event)

    mongo.close()

def events_from_mongo(mstid_list,list_sDate=None,list_eDate=None,months=None,
        category=None,process_level='music',recompute=False,
        db_name='mstid',mongo_port=27017,**kwargs):
    """Allow connection to mongo database."""

    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]

    process_level = more_music.ProcessLevel(str(process_level))

    query_dict  = {'date':{}}
    if list_sDate is not None:
        query_dict['date']['$gte'] = list_sDate

    if list_eDate is not None:
        query_dict['date']['$lt']  = list_eDate

    if list_sDate is not None and list_eDate is not None:
        crsr    = db[mstid_list].find(query_dict)
    else:
        crsr    = db[mstid_list].find()

    if category is not None:
        # Make category iterable.
        category = np.array(category)
        if category.shape == ():
            category.shape = (1,)
        category    = category.tolist()

    event_list  = []
    for item in crsr:
        if category is not None:
            # Allow for a list of categories.
            cat_good = False
            for cat in category:
                this_cat = item.get('category_manu')
                if cat.lower() == 'unclassified':
                    if this_cat is not None: continue
                if cat.lower() == 'none':
                    if this_cat != 'none': continue
                if cat.lower() == 'mstid':
                    if this_cat != 'mstid': continue
                if cat.lower() == 'quiet':
                    if this_cat != 'quiet': continue

                cat_good = True
                break

            if not cat_good: continue

        if months is not None:
            if item['sDatetime'].month not in months:
                continue

        tmp = {}
        tmp['radar']            = str(item['radar'])
        tmp['sTime']            = item['sDatetime']
        tmp['eTime']            = item['fDatetime']
        tmp['category']         = category
        tmp['process_level']    = process_level
        tmp['mstid_list']       = mstid_list
        tmp['db_name']          = db_name
        tmp['mongo_port']       = mongo_port

        tmp                 = dict(tmp,**kwargs)
        event_list.append(tmp)

    event_list  = sorted(event_list,key=lambda k: k['sTime'])
    
    if not recompute:
        # If we want to use what has already been computed, add events to a new
        # list if they need to be computed.  Then make that the actual event_list.
        event_list_1 = []
        for event in event_list:
            completed_process_level = more_music.get_process_level(**event)
            if completed_process_level < process_level:
                event_list_1.append(event)
            else:
                print('events_from_mongo() - SKIPPING - {} {} {!s} {!s}'.format(mstid_list,event['radar'],event['sTime'],event['eTime']))
        event_list = event_list_1

    mongo.close()
    return event_list

def get_mstid_value(mongo_item,sig_key,lambda_max=750,azm_lim=None):
    signals = mongo_item.get('signals')
    if signals is None:
        return 
    if len(signals) == 0: 
        return 

    sig_df  = pd.DataFrame(signals)

    # Get the strongest detected MSTID that has wavelength < lambda_max.
    if lambda_max is not None:
        sig_df = sig_df[sig_df.loc[:,'lambda'] <= lambda_max]

    if azm_lim is not None:
        tf =  np.logical_and( (sig_df['azm']%360. >= azm_lim[0]) ,(sig_df['azm']%360. <= azm_lim[1]) )
        sig_df = sig_df[tf]

    if len(sig_df) == 0: 
        return
    idx = sig_df['max'].idxmax()
    sig = sig_df.loc[idx]

    val = sig.loc[sig_key]
    if sig_key == 'azm':
        val = val % 360.

    return val

def get_mstid_scores(sDate=None,eDate=None,
        mstid_list_format='guc_{radar}_{sDate}_{eDate}',
        db_name='mstid',mongo_port=27017,**kwargs):
    """
    Returns a score for how many radars saw MSTIDs in a given day.  All default
    classified MSTID lists are used. Each radar and analysis window contributes
    to the score, such that:

        mstid --> +1
        quiet --> -1
        None  --> +0

    """
    from . import run_helper #Needs to be imported here to avoid infinite loop import.

    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]

    mstid_lists         = run_helper.get_all_default_mstid_lists(mstid_format=mstid_list_format)

    score_dict  = {}

    for mstid_list in mstid_lists:
        crsr = db[mstid_list].find()
        for item in crsr:
            dt      = item.get('date')
            date    = datetime.datetime(dt.year,dt.month,dt.day)
            score   = score_dict.get(date,0)

            categ = item.get('category_manu')

            if categ == 'mstid':
                score +=  1
            elif categ == 'quiet':
                score += -1

            score_dict[date] = score
    dates,scores    = list(zip(*[(key,val) for key,val in score_dict.items()]))

    df_score        = pd.DataFrame(np.array(scores),index=np.array(dates),
            columns=['score'])
    df_score.sort_index(inplace=True)

    if sDate is not None and eDate is not None:
        tf = np.logical_and(df_score.index >= sDate, df_score.index < eDate)
        df_score = df_score[tf]

    return df_score

def get_mstid_days(sDate=None,eDate=None,
        mstid_list_format='music_guc_{radar}_{sDate}_{eDate}',
        threshold=0.,
        db_name='mstid',mongo_port=27017,**kwargs):
    """
    Returns lists of MSTID and quiet days, based on the daily MSTID score
    and a threshold.
    """

    df_score    = get_mstid_scores(sDate=None,eDate=None,
        mstid_list_format='music_guc_{radar}_{sDate}_{eDate}',
        db_name='mstid',mongo_port=27017,**kwargs)
        
    tf = df_score['score'] <= threshold
    quiet_list  = [x.to_datetime() for x in df_score.index[tf]]

    tf = df_score['score'] > threshold
    mstid_list  = [x.to_datetime() for x in df_score.index[tf]]

    return mstid_list, quiet_list
