"""
tint.objects
============

Functions for managing and recording object properties.

"""

import numpy as np
import pandas as pd
import pyart
from scipy import ndimage
from skimage.measure import regionprops
from skimage.feature import peak_local_max
import datetime as dt
from scipy.stats.mstats import theilslopes

from .grid_utils import get_filtered_frame

def calc_speed_and_dir(df, dx):
    """From a pandas dataframe, use the grid_x, grid_y and time columns to calculate speed (m/s) and 
    direction. Returns both an instantaneous (central finite difference) and average
    (based on linear least squares regression fit to distance/time) speed, as well as 
    direction based on central difference. Note direction is the angle from north that
    the cell is moving TOWARDS"""
    #First check if the storm exists in more than one scan and the uid is not -1 (representing
    # a scan with no storms)
    if (df.shape[0] > 1) & (int(df.reset_index().uid.iloc[0])>=0):
	#Fit a linear polynomial to the distance (from start point) and time of a storm uid.
	#Use the slope to estimate the speed (m/s)
        x, y, t0 = (df["grid_x"].values, df["grid_y"].values, df["time"].iloc[0])
        dist = np.cumsum(np.concatenate(([0], [np.sqrt(np.square(x[i-1] - x[i]) + \
		np.square(y[i-1] - y[i]))  for i in np.arange(1, len(x))]))) 
        times = pd.to_datetime(df["time"]) - t0
        delta_t = (times / np.arange(df.shape[0])).mean().seconds
	#Also estimate the instantaneous speed using a central difference
        speed_inst = np.gradient(dist * dx, delta_t)
        yp = dist * dx
        xp = np.array([t.seconds for t in times])
        fit = theilslopes(yp, xp)
        speed = fit[0]
        speed_rnge = fit[3] - fit[2]

	#Fit a linear polynomial to the x-y grid space for a storm uid.
	#Use the fitted polynomial to estimate the angle of propogation
        fit_xy = np.polyfit(x,y,deg=1)
        yf = np.polyval(fit_xy, x)
        angle_fit = np.arctan2(yf[-1] - yf[0], x[-1] - x[0])
        angle_fit = (90 - ( (180/np.pi) * angle_fit ) + 360) % 360

	#Also calculate the instantaneous angle using a central difference 
        angles_inst = []
        for i in np.arange(df.shape[0]):
            if i == 0:
                angle = (np.arctan2(df["grid_y"].iloc[i+1] - df["grid_y"].iloc[i],\
			df["grid_x"].iloc[i+1] - df["grid_x"].iloc[i]))
            elif i == df.shape[0]-1:
                angle = (np.arctan2(df["grid_y"].iloc[i] - df["grid_y"].iloc[i-1],\
			df["grid_x"].iloc[i] - df["grid_x"].iloc[i-1]))
            else:
                angle = (np.arctan2(df["grid_y"].iloc[i+1] - df["grid_y"].iloc[i-1],\
			df["grid_x"].iloc[i+1] - df["grid_x"].iloc[i-1]))
            angles_inst.append((90 - ( (180/np.pi) * angle ) + 360) % 360)
        df["angle"] = np.round(angle_fit, 3)
        df["angle_inst"] = np.round(angles_inst, 3)
        df["speed"] = np.round(speed, 3)
        df["speed_rnge"] = np.round(speed_rnge, 3)
        df["speed_inst"] = np.round(speed_inst, 3)
    else:
        df["angle"] = np.nan
        df["angle_inst"] = np.nan
        df["speed"] = np.nan
        df["speed_rnge"] = np.nan
        df["speed_inst"] = np.nan
    return df

def num_of_scans(df):
    """For pandas track output, simply append the number of scans in each output as well as the duration"""
    #TODO: Develop for time also? (e.g. rather than just scans)
    if (int(df.reset_index().uid.iloc[0])==-1):
        df["num_of_scans"] = 0
        df["duration_mins"] = np.nan
    else:
        df["num_of_scans"] = df.shape[0]
        df["duration_mins"] = (df.time.max() - df.time.min()).seconds / 60.
    return df

def get_object_center(obj_id, labeled_image):
    """ Returns index of center pixel of the given object id from labeled
    image. The center is calculated as the median pixel of the object extent;
    it is not a true centroid. """
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype('i')
    return center


def get_obj_extent(labeled_image, obj_label):
    """ Takes in labeled image and finds the radius, area, and center of the
    given object. """
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength))/2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {'obj_center': obj_center, 'obj_radius': obj_radius,
                  'obj_area': obj_area, 'obj_index': obj_index}
    return obj_extent


def init_current_objects(first_frame, second_frame, pairs, counter):
    """ Returns a dictionary for objects with unique ids and their
    corresponding ids in frame1 and frame1. This function is called when
    echoes are detected after a period of no echoes. """
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')
    origin = np.array(['-1']*nobj)

    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(first_frame, second_frame,
                                        current_objects)
    return current_objects, counter


def update_current_objects(frame1, frame2, pairs, old_objects, counter):
    """ Removes dead objects, updates living objects, and assigns new uids to
    new-born objects. """
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype='str')
    obs_num = np.array([], dtype='i')
    origin = np.array([], dtype='str')

    for obj in np.arange(nobj) + 1:
        if obj in old_objects['id2']:
            obj_index = old_objects['id2'] == obj
            uid = np.append(uid, old_objects['uid'][obj_index])
            obs_num = np.append(obs_num, old_objects['obs_num'][obj_index] + 1)
            origin = np.append(origin, old_objects['origin'][obj_index])
        else:
            #  obj_orig = get_origin_uid(obj, frame1, old_objects)
            obj_orig = '-1'
            origin = np.append(origin, obj_orig)
            if obj_orig != '-1':
                uid = np.append(uid, counter.next_cid(obj_orig))
            else:
                uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(frame1, frame2, current_objects)
    return current_objects, counter


def attach_last_heads(frame1, frame2, current_objects):
    """ Attaches last heading information to current_objects dictionary. """
    nobj = len(current_objects['uid'])
    heads = np.ma.empty((nobj, 2))
    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0) and
                (current_objects['id2'][obj] > 0)):
            center1 = get_object_center(current_objects['id1'][obj], frame1)
            center2 = get_object_center(current_objects['id2'][obj], frame2)
            heads[obj, :] = center2 - center1
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects['last_heads'] = heads
    return current_objects


def check_isolation(raw, filtered, grid_size, params):
    """ Returns list of booleans indicating object isolation. Isolated objects
    are not connected to any other objects by pixels greater than ISO_THRESH,
    and have at most one peak. """
    nobj = np.max(filtered)
    min_size = params['MIN_SIZE'] / np.prod(grid_size[1:]/1000)
    min_vol = params['MIN_VOL'] / np.prod(grid_size/1000)
    min_height = params['MIN_HGT'] / np.prod(grid_size[0]/1000)
    iso_filtered = get_filtered_frame(raw,
                                      min_size,
                                      min_vol,
                                      min_height,
                                      params['ISO_THRESH'],
                                      params["MIN_FIELD"])
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype='bool')

    for iso_id in np.arange(nobj_iso) + 1:
        obj_ind = np.where(iso_filtered == iso_id)
        objects = np.unique(filtered[obj_ind])
        objects = objects[objects != 0]
        if len(objects) == 1 and single_max(obj_ind, raw, params):
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


def single_max(obj_ind, raw, params):
    """ Returns True if object has at most one peak. """
    max_proj = np.max(raw, axis=0)
    smooth = ndimage.filters.gaussian_filter(max_proj, params['ISO_SMOOTH'])
    padded = np.pad(smooth, 1, mode='constant')
    obj_ind = [axis + 1 for axis in obj_ind]  # adjust for padding
    maxima = 0
    for pixel in range(len(obj_ind[0])):
        ind_0 = obj_ind[0][pixel]
        ind_1 = obj_ind[1][pixel]
        neighborhood = padded[(ind_0-1):(ind_0+2), (ind_1-1):(ind_1+2)]
        max_ind = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
        if max_ind == (1, 1):
            maxima += 1
            if maxima > 1:
                return False
    return True


def get_object_prop(image1, grid1, field, az_field, record, params, steiner):
    """ Returns dictionary of object properties for all objects found in
    image1. 
    """
    id1 = []
    center = []
    grid_x = []
    grid_y = []
    area = []
    dist_min = []
    dist_max = []
    longitude = []
    latitude = []
    field_max = []
    min_height = []
    max_height = []
    eth = []
    volume = []
    local_max = []
    azi_shear = []
    conv_pct = []
    nobj = np.max(image1)

    skimage_props_km1 = ["major_axis_length", "minor_axis_length"]
    skimage_props_km2 = ["area"]

    unit_dim = record.grid_size
    unit_alt = unit_dim[0]/1000
    unit_len = unit_dim[1]/1000
    unit_area = (unit_dim[1]*unit_dim[2])/(1000**2)
    unit_vol = (unit_dim[0]*unit_dim[1]*unit_dim[2])/(1000**3)

    #Set up azimuthal shear 3d array
    raw3D = grid1.fields[field]['data'].data
    if params["AZI_SHEAR"]:
        if az_field in list(grid1.fields.keys()):
           raw3D_az = grid1.fields[az_field]['data'].data
           raw3D_az = np.where(raw3D_az == -9999, np.nan, raw3D_az)
           az_hidx = (np.arange(raw3D.shape[0])*unit_alt >= params["AZH1"]) &\
               (np.arange(raw3D.shape[0])*unit_alt <= params["AZH2"])
        else:
           print("AZI_SHEAR parameter is TRUE, but can't find "+az_field+" in the grid file for"+pyart.util.datetime_from_grid(grid1).strftime())

    #Set up distance array
    x, y = np.meshgrid(grid1.x["data"], grid1.y["data"])
    dist = np.sqrt(np.square(x) + np.square(y))
        
    for obj in np.arange(nobj) + 1:
        obj_index = np.argwhere(image1 == obj)
        id1.append(obj)

        # 2D frame stats
        center.append(np.median(obj_index, axis=0))
        this_centroid = np.round(np.mean(obj_index, axis=0), 3)
        grid_x.append(this_centroid[1])
        grid_y.append(this_centroid[0])
        area.append(obj_index.shape[0] * unit_area)

        rounded = np.round(this_centroid).astype('i')
        cent_met = np.array([grid1.y['data'][rounded[0]],
                             grid1.x['data'][rounded[1]]])

        projparams = grid1.get_projparams()
        lon, lat = pyart.core.transforms.cartesian_to_geographic(cent_met[1],
                                                                 cent_met[0],
                                                                 projparams)

        longitude.append(np.round(lon[0], 4))
        latitude.append(np.round(lat[0], 4))

        # raw 3D grid stats
        obj_slices = [raw3D[:, ind[0], ind[1]] for ind in obj_index]
        field_max.append(np.round(np.nanmax(obj_slices), 3))
        filtered_slices = [obj_slice > params['FIELD_THRESH']
                           for obj_slice in obj_slices]
        filtered_slices_eth = [obj_slice > params['ETH_THRESH']
                           for obj_slice in obj_slices]
        heights = [np.arange(raw3D.shape[0])[ind] for ind in filtered_slices]
        eth_heights = [np.arange(raw3D.shape[0])[ind] for ind in filtered_slices_eth]
        max_height.append(np.max(np.concatenate(heights)) * unit_alt)
        eth.append(np.max(np.concatenate(eth_heights)) * unit_alt)
        min_height.append(np.min(np.concatenate(heights)) * unit_alt)
        volume.append(np.sum(filtered_slices) * unit_vol)
        if params["AZI_SHEAR"]:
            if az_field in list(grid1.fields.keys()):
                azi_shear.append(np.nanpercentile((np.where( (image1==obj) &\
                    (np.nanmax(raw3D, axis=0) >= params["FIELD_THRESH"]),\
                   (np.nanmax(np.abs(raw3D_az[az_hidx]), axis=0)), np.nan)), 99.5))
            else:
                azi_shear.append(np.nan)
        else:
            azi_shear.append(np.nan)

	#Get the number of local maxima using the column maximum reflectivity, 
        #Local maxima must be MIN_DISTANCE (in pixels) apart
        crop = np.where(image1==obj, np.nanmax(raw3D,axis=0), 0)
        local_max_inds = peak_local_max( ndimage.filters.gaussian_filter(crop, params["ISO_SMOOTH"]),
              indices=True,
              exclude_border=0,
              min_distance=params["LOCAL_MAX_DIST"])
        local_max.append(len(local_max_inds))

        #Get maximum and minimum distance of the object from the radar
        obj_dist = np.where(image1==obj, dist, np.nan)/1000.
        obj_dist_min, obj_dist_max = (np.nanmin(obj_dist), np.nanmax(obj_dist))
        dist_min.append(obj_dist_min)
        dist_max.append(obj_dist_max)

        if params["STEINER"]:
            steiner_slice = np.where(image1==obj, steiner, np.nan)
            if (steiner_slice>=1).sum() == 0:
                conv_pct.append(np.nan)
            else:
                conv_pct.append((steiner_slice==2).sum() / (steiner_slice>=1).sum())
        else:
            conv_pct.append(np.nan)

    # cell isolation
    isolation = check_isolation(raw3D, image1, record.grid_size, params)

    objprop = {'id1': id1,
               'center': center,
               'grid_x': grid_x,
               'grid_y': grid_y,
               'area_km': area,
               'dist_min': dist_min,
               'dist_max': dist_max,
               'field_max': field_max,
               'min_height': min_height,
               'max_height': max_height,
               'eth': eth,
               'volume': volume,
               'lon': longitude,
               'lat': latitude,
               'isolated': isolation,
               'local_max': local_max,
               'conv_pct': conv_pct,
               'azi_shear': azi_shear}

    if params['SKIMAGE_PROPS']:
        rp = regionprops(image1, intensity_image=raw3D.max(axis=0))
        for p in params['SKIMAGE_PROPS']:
            if p in skimage_props_km1:
                objprop[p] = [r[p] * unit_len for r in rp]
            elif p in skimage_props_km2:
                objprop[p] = [r[p] * unit_area for r in rp]
            else:
                objprop[p] = [r[p] for r in rp]

    return objprop


def write_tracks(old_tracks, record, current_objects, obj_props, params):
    """ Writes all cell information to tracks dataframe. """

    nobj = len(obj_props['id1'])
    scan_num = [record.scan] * nobj
    uid = current_objects['uid']

    new_tracks = pd.DataFrame({
        'scan': scan_num,
        'uid': uid,
        'time': record.time,
        'grid_x': obj_props['grid_x'],
        'grid_y': obj_props['grid_y'],
        'lon': obj_props['lon'],
        'lat': obj_props['lat'],
        'area_km': obj_props['area_km'],
        'dist_min': np.round(obj_props['dist_min'],3),
        'dist_max': np.round(obj_props['dist_max'],3),
        'vol': obj_props['volume'],
        'field_max': obj_props['field_max'],
        'min_alt': obj_props['min_height'],
        'max_alt': obj_props['max_height'],
        'eth': obj_props['eth'],
        'isolated': obj_props['isolated'],
        'local_max': np.round(obj_props['local_max'], 3),
        'azi_shear': np.round(obj_props['azi_shear'], 3),
        'conv_pct': np.round(obj_props['conv_pct'], 3),
    })
    if params['SKIMAGE_PROPS']:
        for p in params['SKIMAGE_PROPS']:
            try:
               new_tracks[p] = np.round(obj_props[p], 3)
            except:
               new_tracks[p] = obj_props[p]

    new_tracks.set_index(['scan', 'uid'], inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks

def write_null_tracks(old_tracks, record):
    """ If there is no objects in a scan, write null output. """

    scan_num = record.scan
    uid = -1

    new_tracks = pd.DataFrame({
        'scan': scan_num,
        'uid': [uid],
        'time': record.time,
        'grid_x': np.nan,
        'grid_y': np.nan,
        'lon': np.nan,
        'lat': np.nan,
        'area_km': np.nan,
        'dist_min': np.nan,
        'dist_max': np.nan,
        'vol': np.nan,
        'field_max': np.nan,
        'min_alt': np.nan,
        'max_alt': np.nan,
        'eth': np.nan,
        'isolated': np.nan,
        'local_max': np.nan,
        'azi_shear': np.nan,
        'conv_pct': np.nan
    })

    new_tracks.set_index(['scan', 'uid'], inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks
