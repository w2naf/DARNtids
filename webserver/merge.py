import copy
import numpy as np
import pydarn.proc.music as music
import scipy as sp

def merge_data(dataObj_0,dataObj_1,dataSet_0='active',dataSet_1='active',interp=True):
    if interp:
        new_dataObj = copy.deepcopy(dataObj_0)

        dataSet_0   = music.getDataSet(new_dataObj,dataSet_0)
        dataSet_1   = music.getDataSet(dataObj_1,dataSet_1)

        #Check for matching times in both data sets.
        new_times   = [x for x in dataSet_0.time if x in dataSet_1.time]

        lat_0 = dataSet_0.fov.latCenter
        lat_1 = dataSet_1.fov.latCenter
        lon_0 = dataSet_0.fov.lonCenter
        lon_1 = dataSet_1.fov.lonCenter

        lat_min     = np.min([np.nanmin(lat_0),np.nanmin(lat_1)])
        lat_max     = np.max([np.nanmax(lat_0),np.nanmax(lat_1)])
        lon_min     = np.min([np.nanmin(lon_0),np.nanmin(lon_1)])
        lon_max     = np.max([np.nanmax(lon_0),np.nanmax(lon_1)])

        step_sz     = 0.25
    #    grid_lat,grid_lon           = np.mgrid[lat_min:lat_max:step_sz, lon_min:lon_max:step_sz]
    #    grid_lat_full,grid_lon_full = np.mgrid[lat_min-0.5*step_sz:lat_max+0.5*step_sz:step_sz, lon_min-0.5*step_sz:lon_max+0.5*step_sz:step_sz]

        grid_lon,grid_lat           = np.mgrid[lon_min:lon_max:step_sz,lat_min:lat_max:step_sz]
        grid_lon_full,grid_lat_full = np.mgrid[lon_min-0.5*step_sz:lon_max+0.5*step_sz:step_sz,lat_min-0.5*step_sz:lat_max+0.5*step_sz:step_sz]
        new_nr_beams,new_nr_gates   = grid_lon.shape

        lat_vec     = np.concatenate([lat_0.flatten(),lat_1.flatten()])
        lon_vec     = np.concatenate([lon_0.flatten(),lon_1.flatten()])

        lat_good        = np.logical_not(np.isnan(lat_vec))
        lon_good        = np.logical_not(np.isnan(lon_vec))
        lat_lon_good    = np.logical_and(lat_good,lon_good)

    #    lat_lon_inx     = np.where(lat_lon_good)
    #    lat_lon         = np.array([lat_vec[lat_lon_good],lon_vec[lat_lon_good]]).T

        new_data        = np.zeros([len(new_times),new_nr_beams,new_nr_gates],dtype=np.float)
        for inx,time in enumerate(new_times):
            time_0_inx  = np.where(dataSet_0.time == time)[0][0]
            data_0      = dataSet_0.data[time_0_inx,:,:]

            time_1_inx  = np.where(dataSet_1.time == time)[0][0]
            data_1      = dataSet_1.data[time_1_inx,:,:]

            data_vec_tmp    = np.concatenate([data_0.flatten(),data_1.flatten()])
            data_vec_good   = np.logical_not(np.isnan(data_vec_tmp))

            all_good        = np.logical_and(lat_lon_good,data_vec_good)

            lat_lon         = np.array([lat_vec[all_good],lon_vec[all_good]]).T
            data_vec        = data_vec_tmp[all_good]

            tmp = sp.interpolate.griddata(lat_lon,data_vec,(grid_lat,grid_lon),fill_value=0.)
            new_data[inx,:,:] = tmp


        merged_data = dataSet_0.copy('merged_interp','Merged Interpolated')
        # Merge along beams
        merged_data.data = new_data
        merged_data.time = np.array(new_times)

        #Merge FOV Data
        merged_data.fov.beams           = np.arange(new_nr_beams)
        merged_data.fov.gates           = np.arange(new_nr_gates)
        merged_data.fov.latCenter       = grid_lat
        merged_data.fov.latFull         = grid_lat_full
        merged_data.fov.lonCenter       = grid_lon
        merged_data.fov.lonFull         = grid_lon_full
        merged_data.fov.slantRCenter    = np.arange(new_nr_gates)
        merged_data.fov.slantRFull      = np.arange(new_nr_gates+1)

        merged_data.setActive()
        return new_dataObj
    else:
        new_dataObj = copy.deepcopy(dataObj_0)

        dataSet_0   = music.getDataSet(new_dataObj,dataSet_0)
        dataSet_1   = music.getDataSet(dataObj_1,dataSet_1)
        merged_data = dataSet_0.copy('merged_direct','Merged Direct')

        # Merge along beams
        merged_data.data = np.concatenate([dataSet_0.data,dataSet_1.data],axis=1)

        #Merge FOV Data
        total_beams = dataSet_0.fov.beams.size + dataSet_1.fov.beams.size
        merged_data.fov.beams = np.arange(total_beams)
        merged_data.fov.latCenter = np.concatenate([dataSet_0.fov.latCenter,dataSet_1.fov.latCenter],axis=0)
        merged_data.fov.latFull   = np.concatenate([dataSet_0.fov.latFull  ,dataSet_1.fov.latFull[1:,:]],axis=0)
        merged_data.fov.lonCenter = np.concatenate([dataSet_0.fov.lonCenter,dataSet_1.fov.lonCenter],axis=0)
        merged_data.fov.lonFull   = np.concatenate([dataSet_0.fov.lonFull  ,dataSet_1.fov.lonFull[1:,:]],axis=0)
        merged_data.fov.slantRCenter = np.concatenate([dataSet_0.fov.slantRCenter,dataSet_1.fov.slantRCenter],axis=0)
        merged_data.fov.slantRFull   = np.concatenate([dataSet_0.fov.slantRFull  ,dataSet_1.fov.slantRFull[1:,:]],axis=0)

        merged_data.setActive()
        return new_dataObj
