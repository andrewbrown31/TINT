"""
tint.grid_utils
===============

Tools for pulling data from pyart grids.


"""

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
import tobac


def parse_grid_datetime(grid_obj):
    """ Obtains datetime object from pyart grid_object. """
    dt_string = grid_obj.time['units'].split(' ')[-1]
    date = dt_string[:10]
    time = dt_string[11:19]
    dt0 = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    dt = datetime.timedelta(seconds=float(grid_obj.time['data'][0])) + dt0
    return dt


def get_grid_size(grid_obj):
    """ Calculates grid size per dimension given a grid object. """
    z_len = grid_obj.z['data'][-1] - grid_obj.z['data'][0]
    x_len = grid_obj.x['data'][-1] - grid_obj.x['data'][0]
    y_len = grid_obj.y['data'][-1] - grid_obj.y['data'][0]
    z_size = z_len / (grid_obj.z['data'].shape[0] - 1)
    x_size = x_len / (grid_obj.x['data'].shape[0] - 1)
    y_size = y_len / (grid_obj.y['data'].shape[0] - 1)
    return np.array([z_size, y_size, x_size])


def get_radar_info(grid_obj):
    radar_lon = grid_obj.radar_longitude['data'][0]
    radar_lat = grid_obj.radar_latitude['data'][0]
    info = {'radar_lon': radar_lon,
            'radar_lat': radar_lat}
    return info


def get_grid_alt(grid_size, alt_meters=1500):
    """ Returns z-index closest to alt_meters. """
    return np.int(np.round(alt_meters/grid_size[0]))


def get_vert_projection(grid, thresh=40):
    """ Returns boolean vertical projection from grid. """
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(grid, min_size, min_vol, min_height, thresh):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    echo_height = get_vert_projection(grid, thresh)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, grid, min_size, min_vol, min_height, thresh)
    return frame


def clear_small_echoes(label_image, grid, min_size, min_vol, min_height, thresh):
    """ Takes in binary image and clears objects less than min_size, min_height and min_vol. """
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0

    for obj in np.unique(label_image):
        if obj > 0:
            heights = (np.where(grid[:,label_image==obj] > thresh))[0]
            if (((grid[:,label_image == obj] > thresh).sum()) < min_vol) | \
                         ((heights.max() - heights.min()) < min_height):
                label_image[label_image == obj] = 0

    #label_image = ndimage.label(label_image)
    #return label_image[0]
    cnt=1
    for obj in np.unique(label_image):
        if obj > 0:
            label_image[label_image==obj] = cnt
            cnt=cnt+1
    return label_image

def extract_grid_data(grid_obj, field, grid_size, params):
    """ Returns filtered grid frame and raw grid slice at global shift
    altitude. """
    min_size = params['MIN_SIZE'] / np.prod(grid_size[1:]/1000)
    min_vol = params['MIN_VOL'] / np.prod(grid_size/1000)
    min_height = params['MIN_HGT'] / np.prod(grid_size[0]/1000)
    masked = grid_obj.fields[field]['data']
    #masked.data[masked.data == masked.fill_value] = 0
    masked[masked.mask] = 0
    gs_alt = params['GS_ALT']
    raw = masked.data[get_grid_alt(grid_size, gs_alt), :, :]

    if params["SEGMENTATION_METHOD"] == "thresh":
        frame = get_filtered_frame(masked.data, min_size, min_vol, min_height,
                               params['FIELD_THRESH'])

    elif params["SEGMENTATION_METHOD"] == "watershed":

	#Instead of labelling the 2D grid using scipy ndimage (which just gets > THRESH
	#   objects) with get_filtered_frame(), use watershedding from tobac
	#This requires a conversion from the pyart grid object to an Iris cube
        colmax = grid_obj.to_xarray()[field].max("z")
        for n in colmax.coords:
            if n not in ["time","x","y"]:
                colmax = colmax.drop(n)
        colmax = colmax.where(~colmax.isnull(),0).to_iris()
        
        #Set the segmentation and feature ID parameters for tobac. This will need to 
        # be set as an option in the TINT parameter set, eventually.
        parameters_features={}
        parameters_segmentation={}
        parameters_features['position_threshold']='weighted_diff'
        parameters_features['sigma_threshold']=params["WATERSHED_SMOOTHING"]
        parameters_features['threshold']=params["WATERSHED_THRESH"]
        parameters_features['threshold']=params["WATERSHED_THRESH"]
        parameters_features['n_erosion_threshold']=params["WATERSHED_EROSION"]
        parameters_segmentation['threshold']=params["FIELD_THRESH"]
        
        #Run the tobac feature detection
        Features=tobac.feature_detection.feature_detection_multithreshold(colmax,grid_size[1],**parameters_features)
        if Features is None:
           frame = np.zeros(masked.data.shape)
        else:
           #Segmentation using tobac based on a single threshold and Feature locations
           Mask_refl,Features=tobac.segmentation.segmentation(Features,colmax,grid_size[1],**parameters_segmentation)
           #Clear small objects
           frame=clear_small_echoes(Mask_refl.data, masked.data, params["MIN_SIZE"], params["MIN_VOL"], params["MIN_HGT"], params['FIELD_THRESH'])
        
    else:

        raise ValueError("SEGMENTATION METHOD "+params["SEGMENTATION_METHOD"]+" IS NOT VALID. SHOULD BE thresh OR watershed")

    return raw, frame
