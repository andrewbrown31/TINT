#file created to work in sync with tint code to write the desired h5 files. 
#created by Stacey Hitchcock 2019
import numpy as np
import h5py
import datetime
import math

def Setup_h5File(grid1,outdir):
    #TODO EDIT GLOABL ATTRS
    savef=h5py.File(outdir,'w')
    savef.attrs['proj']=np.string_('pyart_aeqd')
    savef.attrs['orig_field_name']=np.string_('reflectivity_cal_cor')
    savef.attrs['product_fields_info']=np.string_('name: reflectivity volume  units:dBz,\
                                             name: cell_mask units: binary')
    savef.attrs['lon_0']=grid1.origin_longitude['data'][0]
    savef.attrs['lat_0']=grid1.origin_latitude['data'][0]
    savef.attrs['nx']=grid1.nx
    savef.attrs['ny']=grid1.ny
    savef.attrs['dx']='1 km'
    savef.attrs['dy']='1 km'
    savef.attrs['x_minmax']=grid1.x['data'].min(),grid1.x['data'].max()
    savef.attrs['y_minmax']=grid1.y['data'].min(),grid1.y['data'].max()
    savef.attrs['outergroup']=np.string_('Time, formatted %H%M%S')
    savef.attrs['innergroup']=np.string_('Unique object ID (uid)')
    savef.attrs['created']=np.string_(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'))
    savef.attrs['source']=np.string_('Australian Open Radar Dataset V1.0 Level 1b data')
    savef.attrs['acknowlegement']=np.string_('This work is supported by the ARC Centre for \
          Excellence in Climate Extremes using a modified version of TINT \
          (https://github.com/openradar/TINT). Thanks Jordan Brook and Joshua Soderholm \
          for guidence and original dataset. Thanks Todd Lane for support.')
    savef.attrs['creator_name']=np.string_('Stacey Hitchcock')
    savef.attrs['creator_email']=np.string_('Stacey.Hitchcock@unimelb.edu.au')
    savef.attrs['host']=np.string_('NCI - National Computing Infrastructure')
    savef.attrs['site_name']=np.string_('Melb') #will need to be modified if you expand
    savef.attrs['state']=np.string_('VIC')
    savef.attrs['country']=np.string_('Australia')

    #Save grid lat, lon as level 1 groups, for reconstruction later
    lon, lat = grid1.get_point_longitude_latitude()
    lat_group = savef.create_group("lat")
    lon_group = savef.create_group("lon")
    lat_group.create_dataset('lat',
			    shape=lat.shape,
			    data=lat)
    lon_group.create_dataset('lon',
			    shape=lon.shape,
			    data=lon)
    return savef

def write_griddata(savef,image1,grid1,field,current_objects,record,obj_props):
    print('Writing h5 files for scan', record.scan)

    nobj = len(obj_props['id1'])
    scan_num = [record.scan] * nobj
    uids = current_objects['uid']
    dts=record.time
    outtime=dts.strftime('%Y%m%d%H%M%S')
    ttgroup=savef.create_group(outtime)
    for IN in np.arange(nobj):
        objIN=obj_props['id1'][IN]
        uid=uids[IN]
        uidgroup=ttgroup.create_group(uid)
        boxIN=np.squeeze(obj_props['bbox'][IN])
        cellimage=image1[boxIN[0]:boxIN[2],boxIN[1]:boxIN[3]] 
        cellmasked=np.ma.masked_equal(cellimage,objIN)#careful in reconstruction. other cells may show as 0s in the frame
        cellmask=cellmasked.mask
        cellgriddims=cellmask.shape
        writecellmask=uidgroup.create_dataset('cell_mask',
                                              shape=cellgriddims,
                                              data=cellmask.astype(int))      
        uidgroup.attrs["bbox"] = boxIN
        uidgroup['cell_mask'].dims[0].label='cell_y'
        uidgroup['cell_mask'].dims[1].label='cell_x'
        
	#To reconstruct, use this code
#	f = h5py.File(outdir, "r")
#	group_id = "20151215230102/2/"
#       bbox = f[group_id].attrs["bbox"]
#	x = f["lon/lon"][:]
#	y = f["lat/lat"][:]
#       recon = np.zeros(x.shape)
#       recon[bbox[0]:bbox[2],bbox[1]:bbox[3]] = f[group_id+"/cell_mask"][:]
#       plt.pcolormesh(x,y,recon)

    return savef

