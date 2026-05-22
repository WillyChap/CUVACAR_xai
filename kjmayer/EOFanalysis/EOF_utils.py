import numpy as np
import glob
import xarray as xr

def load_subsetdata(lat, lon, lev, lat_of_interest, lon_of_interest, lev_of_interest):
    # Find associated index value for getting data
    lat_of_interest_near = lat.sel(lat=lat_of_interest, method='nearest')
    lon_of_interest_near = lon.sel(lon=lon_of_interest, method='nearest')
    lev_of_interest_near = lev.sel(lev=lev_of_interest, method='nearest')
    
    ilat_of_interest = np.where(lat.values == lat_of_interest_near.values)[0][0]
    ilon_of_interest = np.where(lon.values == lon_of_interest_near.values)[0][0]
    ilev_of_interest = np.where(lev.values == lev_of_interest_near.values)[0][0]
    
    ilat_of_interest_str = f"{ilat_of_interest:05}"
    ilon_of_interest_str = f"{ilon_of_interest:05}"
    ilev_of_interest_str = f"{ilev_of_interest:05}"

    # grab all dates for the U level, lat, and lon point
    dir_o = "/glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/"
    files = sorted(
        glob.glob(dir_o+"*/tensor_steps24_parrallel_lev"+ilev_of_interest_str+"_lat"+ilat_of_interest_str+"_lon"+ilon_of_interest_str+".npy")
    )
    files_JJA = files[151:243]
    files_DJF = np.delete(files, np.s_[59:334], axis=0) #remove non-DJF
    
    IG_allvars_JJA = np.stack([np.load(f).squeeze() for f in files_JJA])
    IG_allvars_DJF = np.stack([np.load(f).squeeze() for f in files_DJF])

    return IG_allvars_JJA, IG_allvars_DJF, ilat_of_interest, ilon_of_interest


def spatial_cube_subset(IG_allvars, addval, lat, lon, lat_range_of_interest, lon_range_of_interest, ilat_of_interest, ilon_of_interest):
    # Preprocess Data for 3D EOF
    delta_lat = float((lat[1] - lat[0]).values)
    delta_lon = float((lon[1] - lon[0]).values)

    south_lat_of_interest = int(ilat_of_interest-(lat_range_of_interest/delta_lat))
    north_lat_of_interest = int(ilat_of_interest+(lat_range_of_interest/delta_lat))
    
    west_lon_of_interest = int(ilon_of_interest-(lon_range_of_interest/delta_lon))
    east_lon_of_interest = int(ilon_of_interest+(lon_range_of_interest/delta_lon))

    if west_lon_of_interest < 0:
        IG_allvars_cubesubset_temp = np.delete(IG_allvars, np.s_[east_lon_of_interest:west_lon_of_interest], axis=-1)
        IG_allvars_cubesubset = IG_allvars_cubesubset_temp[:, 0+addval:32+addval, south_lat_of_interest:north_lat_of_interest]

    else:
        IG_allvars_cubesubset = IG_allvars[:, 0+addval:32+addval, south_lat_of_interest:north_lat_of_interest, west_lon_of_interest:east_lon_of_interest]
    
    return IG_allvars_cubesubset


def eof_func(X, num_eofs=3):
    # Remove time mean
    X_anom = X - np.mean(X, axis=0, keepdims=True)

    # Covariance in vertical/space dimension
    C = np.cov(X_anom, rowvar=False)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Leading EOF and PC
    eof = eigvecs[:, :num_eofs]
    pc = X_anom @ eof

    return eof, pc, eigvals[:num_eofs] / np.sum(eigvals)


def make_eof_ds(eof_reshape, pc, eigvals, lats, lons, levs, eof_nums):
    return xr.Dataset(
        {
            "eof_map": xr.DataArray(
                eof_reshape,
                dims=["lev", "lat", "lon", "eof"],
                coords={"lev": levs, "lat": lats, "lon": lons, "eof": eof_nums},
            ),
            "pc": xr.DataArray(
                pc,                          # shape (time, 3)
                dims=["time", "eof"],
                coords={"eof": eof_nums},
            ),
            "eigenvalues": xr.DataArray(
                eigvals,                     # shape (3,)
                dims=["eof"],
                coords={"eof": eof_nums},
            ),
        }
    )