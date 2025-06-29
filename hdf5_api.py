# Nicholas Guerra
# Graduate Thesis in Software Engineering
# Spring 2024 - Spring 2025
# Dr. Nathaniel A. Frissell, Professor Paul M. Jackowitz

import h5py
import numpy as np
import datetime
from pyDARNmusic.music.music_array import musicArray
from pyDARNmusic.music.music_data_object import musicDataObj
from pyDARNmusic.music.signals_detected import SigDetect

def formatData(obj):
    """
    Recursively format objects to their needed types for HDF5 storage.
    """
    # Datetime class; return string representation of the datetime class.
    if obj is datetime.datetime:
        return "datetime.datetime"
    # Datetime instances; return their ISO string representation.
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    # Scalars; return the data as is.
    elif isinstance(obj, (int, float, str)):
        return obj
    # Lists & Tuples: recursively format items into a list.
    elif isinstance(obj, (list, tuple)):
        return [formatData(item) for item in obj]
    # Dicts; recursively format items into a dict.
    elif isinstance(obj, dict):
        return {formatData(key): formatData(value) for key, value in obj.items()}
    # Otherwise, return string representation.
    else:
        return str(obj)

def saveDictToHDF5(hdf5Group, dictionary):
    """
    Save the contents of a dictionary to an HDF5 group.
    """
    for key, values in dictionary.items():
        try:
            # Skip the "parent" key.
            if key == "parent":
                continue

            # SigDetect dictionary saved as its own subgroup with individual datasets.
            if key == "sigDetect":
                sigDetectGroup = hdf5Group.create_group("sigDetect")
                if values.info:
                    # All possible keys in a SigDetect info attribute.
                    expectedSDKeys = {
                        'labelInx', 'order', 'area', 'max', 'maxpos', 'kx', 'ky', 'k', 
                        'lambda_x', 'lambda_y', 'lambda', 'azm', 'freq', 'period', 'vel'
                    }
                    # Process of identifying existing keys in the info attribute, sorting them, 
                    # and storing them as a properly formatted dataset.
                    cleanedInfo = []
                    for sig in values.info:
                        filteredSignal = {k: sig[k] for k in expectedSDKeys if k in sig}
                        cleanedInfo.append(filteredSignal)
                    sortedKeys = sorted(expectedSDKeys)
                    dtypeFormatted = np.dtype([
                        (k, np.float64 if k not in {'labelInx', 'order', 'maxpos'} 
                         else ('2i' if k == 'maxpos' else np.int32))
                        for k in sortedKeys
                    ])
                    infoArray = np.array(
                        [tuple(sig[k] for k in sortedKeys) for sig in cleanedInfo],
                        dtype=dtypeFormatted
                    )
                    sigDetectGroup.create_dataset("info", data=infoArray)
                sigDetectGroup.create_dataset("labels", data=values.labels.astype(np.int32))
                sigDetectGroup.create_dataset("mask", data=values.mask.astype(np.uint8))
                sigDetectGroup.create_dataset("nrSigs", data=np.int32(values.nrSigs))
                continue

            # Store values that are the datetime class (from diffs, time, metadata, history attributes) as a string.
            if values is datetime.datetime:
                hdf5Group.create_dataset(key, data=np.bytes_("datetime.datetime"))
                continue

            if key == "fov":
                if key not in hdf5Group:
                    # Save datetime instances as numpy ISO strings directly within a dataset.
                    if isinstance(values, datetime.datetime):
                        hdf5Group.create_dataset(key, data=np.bytes_(values.isoformat()))
                        continue
                    # Save lists as numpy arrays directly within a dataset.
                    if isinstance(values, list):
                        hdf5Group.create_dataset(key, data=np.array(values))
                    # Save numpy N-dimensional arrays directly within a dataset.
                    elif isinstance(values, np.ndarray):
                        hdf5Group.create_dataset(key, data=values)
                    # Save dictionaries as a subgroup and save their contents.
                    elif isinstance(values, dict):
                        subGroup = hdf5Group.create_group(key)
                        saveDictToHDF5(subGroup, values)
                    # Save scalars directly within a dataset (or as a numpy string if the values are strings).
                    elif isinstance(values, (int, float, str)):
                        hdf5Group.create_dataset(key, data=np.bytes_(values) if isinstance(values, str) else values)
            # Save datetimes as datasets composed of formatted numpy strings.
            elif isinstance(values, datetime.datetime):
                hdf5Group.create_dataset(key, data=np.bytes_(formatData(values)))
                continue
            # Save lists of datetimes as datasets composed of numpy arrays of formatted datetimes.
            elif isinstance(values, list) and all(isinstance(v, datetime.datetime) for v in values):
                hdf5Group.create_dataset(key, data=np.array([formatData(v) for v in values], dtype='S'))
                continue
            # Save dictionaries as subgroups, format their data, and save the dictionary's contents to the subgroup.
            elif isinstance(values, dict):
                subGroup = hdf5Group.create_group(key)
                formattedValues = formatData(values)
                saveDictToHDF5(subGroup, formattedValues)
            # Save lists consisting entirely of ints/np ints or floats/np floats as datasets composed of a numpy array 
            # of those values, and saves generic lists as datasets composed of numpy arrays of formatted values. 
            elif isinstance(values, list):
                if values and all(isinstance(x, (int, float, np.integer, np.floating)) for x in values):
                    hdf5Group.create_dataset(key, data=np.array(values))
                else:
                    formattedValues = formatData(values)
                    hdf5Group.create_dataset(key, data=np.array(formattedValues, dtype='S'))
            # Save numpy N-dimensional arrays as datasets variously.
            elif isinstance(values, np.ndarray):
                if values.dtype == 'O':
                    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in values):
                        hdf5Group.create_dataset(key, data=values.astype(float))
                    else:
                        formattedValues = np.array([formatData(item) for item in values], dtype='S')
                        hdf5Group.create_dataset(key, data=formattedValues)
                else:
                    hdf5Group.create_dataset(key, data=values)
            # Saves bools, ints, and floats directly within datasets.
            elif isinstance(values, (bool, int, float)):
                hdf5Group.create_dataset(key, data=values)
            # Otherwise, saves the data in its own dataset composed of numpy strings of formatted values.
            else:
                formattedValue = formatData(values)
                hdf5Group.create_dataset(key, data=np.bytes_(formattedValue))
        except Exception as e:
            print(f"Could not save {key} in {hdf5Group.name}: {e}")

def saveMusicArrayToHDF5(musicArrayObj, filename):
    """
    Save the contents of a musicArray object ('DS###_*', 'active', 'prm' and 'messages' attributes) to HDF5.
    """
    with h5py.File(filename, 'w') as hdf5File:
        for attributeName in dir(musicArrayObj):
            # Skip built-in attributes.
            if attributeName.startswith('__'):
                continue
            attributeValue = getattr(musicArrayObj, attributeName)
            # Create an HDF5 group for dict attributes (prm) and save its contents.
            if isinstance(attributeValue, dict):
                group = hdf5File.create_group(attributeName)
                saveDictToHDF5(group, attributeValue)
            # Store list attributes (messages) as numpy arrays of strings.
            elif isinstance(attributeValue, list):
                hdf5File.create_dataset(attributeName, data=np.array(attributeValue, dtype='S'))
            # Create individual HDF5 groups for 'DS' and 'active' attributes and save their __dict__'s.
            elif attributeName.startswith('DS') or attributeName.startswith('active'):
                dsGroup = hdf5File.create_group(attributeName)
                saveDictToHDF5(dsGroup, attributeValue.__dict__)

def loadMusicArrayFromHDF5(hdf5FilePath):
    """
    Reconstruct a musicArray object from an HDF5 file.
    """
    with h5py.File(hdf5FilePath, 'r') as hdf5File:
        # Extract metadata from first level of processing.
        try:
            metadata = hdf5File['DS000_originalFit/metadata']
        except KeyError:
            # The file does not have metadata, i.e. "No data for this time period" in messages dataset.
            return None
            
        sTime         = metadata['sTime']
        eTime         = metadata['eTime']
        param         = metadata['param']
        gscat         = metadata['gscat']
        fovModel      = metadata['model']
        fovElevation  = metadata['elevation']
        fovCoords     = metadata['coords']
        channel       = metadata['channel']
        file_type     = metadata['fType']

        reconstructedMusicArray = musicArray(
            fitacf=None,
            sTime=sTime,
            eTime=eTime,
            param=param,
            gscat=gscat,
            fovElevation=fovElevation,
            fovModel=fovModel,
            fovCoords=fovCoords,
            channel=channel,
            file_type=file_type
        )

        # Iterate over hdf5 file's 'DS' and 'active' keys, extract their data, and save the data to the 
        # newly reconstructed musicArray as a new musicDataObj. Otherwise, save the data to the newly 
        # reconstructed musicArray directly.
        for key in hdf5File.keys():
            if key.startswith("DS") or key.startswith("active"):
                dsGroup = hdf5File[key]
                newMusicDataObj = musicDataObj(time=None, data=None, parent=reconstructedMusicArray)
                for subkey in dsGroup.keys():
                    if subkey == "sigDetect":
                        newMusicDataObj.sigDetect = loadSigDetectFromHDF5(dsGroup[subkey])
                    else:
                        newMusicDataObj.__dict__[subkey] = extractDataFromHDF5(dsGroup[subkey])
                setattr(reconstructedMusicArray, key, newMusicDataObj)
            else:
                setattr(reconstructedMusicArray, key, extractDataFromHDF5(hdf5File[key]))
    return reconstructedMusicArray

def convertToUnicode(data):
    """
    If inputted data is a UTF-8 encoded byte string, convert to Unicode.
    Try to convert data into a datetime object. If conversion fails, return the data.
    """
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    try:
        return datetime.datetime.fromisoformat(data)
    except ValueError:
        return data

def extractDataFromHDF5(hdf5Item):
    """
    Recursively extract data from an HDF5 group or dataset.
    """
    if isinstance(hdf5Item, h5py.Group):
        # Store groups as dictionaries.
        dictionary = {}
        for key in hdf5Item.keys():
            dictionary[key] = extractDataFromHDF5(hdf5Item[key])
        return dictionary
    elif isinstance(hdf5Item, h5py.Dataset):
        # Store datasets as various types (tuples, lists, integers, numpy arrays, or default types.)
        try:
            data = hdf5Item[()]
            if hdf5Item.name.endswith("timeLimits"):
                if isinstance(data, np.ndarray) and data.dtype.kind in {'S', 'U'}:
                    data = data.astype(str)
                    return tuple(datetime.datetime.fromisoformat(x) for x in data)
                if isinstance(data, list):
                    return tuple(datetime.datetime.fromisoformat(x) for x in data)
            if hdf5Item.name.endswith("rangeLimits") or hdf5Item.name.endswith("gateLimits"):
                if isinstance(data, np.ndarray):
                    if data.dtype.kind in {'i', 'u', 'f'}:
                        return data.tolist()
                    elif data.dtype.kind in {'S', 'U'}:
                        return list(map(int, data.astype(str)))
            try:
                if data.isdigit():
                    return int(data)
            except Exception:
                pass
            if isinstance(data, np.ndarray) and data.dtype.kind in {'S', 'U'}:
                return np.array([convertToUnicode(x) for x in data])
            return data
        except Exception as e:
            print(f"Error reading dataset {hdf5Item.name}: {e}")
            return None

def loadSigDetectFromHDF5(sigDetectGroup):
    """
    Reconstruct a SigDetect object from an HDF5 group.
    """
    sigDetect = SigDetect()
    # Check if each attribute that makes up a SigDetect object is within the
    # HDF5 sigDetect group and save it to the sigDetect object if so.
    if "info" in sigDetectGroup:
        infoArray = sigDetectGroup["info"][()]
        sigDetect.info = [dict(zip(infoArray.dtype.names, row)) for row in infoArray]
    if "labels" in sigDetectGroup:
        sigDetect.labels = sigDetectGroup["labels"][()].astype(np.int32)
    if "mask" in sigDetectGroup:
        sigDetect.mask = sigDetectGroup["mask"][()].astype(bool)
    if "nrSigs" in sigDetectGroup:
        sigDetect.nrSigs = int(sigDetectGroup["nrSigs"][()])
    return sigDetect
