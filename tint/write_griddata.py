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
    savef.attrs['lon_0']=grid1.origin_longitude['data'][0]
    savef.attrs['lat_0']=grid1.origin_latitude['data'][0]
    savef.attrs['nx']=grid1.nx
    savef.attrs['ny']=grid1.ny
    savef.attrs['x_minmax']=grid1.x['data'].min(),grid1.x['data'].max()
    savef.attrs['y_minmax']=grid1.y['data'].min(),grid1.y['data'].max()
    savef.attrs['outergroup']=np.string_('Time, formatted %H%M%S')
    savef.attrs['innergroup']=np.string_('Unique object ID (uid)')
    savef.attrs['created']=np.string_(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'))
    savef.attrs['source']=np.string_('Australian Open Radar Dataset V1.0 Level 1b data')
    savef.attrs['creator_name']=np.string_('Andrew Brown')
    savef.attrs['creator_email']=np.string_('andrewb1@student.unimelb.edu.au')
    savef.attrs['host']=np.string_('NCI - National Computing Infrastructure')
    #TODO
    #The following line doesn't work. Probably need to loop over grid1 metadata keys and append to savef
    for key in grid1.metadata.keys():
        savef.attrs["source_"+key] = grid1.metadata[key]

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

	#To get group_id from csv track output
	#df["group_id"] = pd.DatetimeIndex(df["time"]).strftime("%Y%m%d%H%M%S") + "/" + df["uid"].astype(str)

    return savef

