import numpy as np
import glob
import xarray as xr


def nearest_coordinate_index(coord, value):
    """Return the nearest index for a nonperiodic coordinate."""
    coord_values = np.asarray(coord.values, dtype=float)
    return int(np.argmin(np.abs(coord_values - value)))


def nearest_longitude_index(lon, value):
    """Return nearest longitude index using cyclic distance."""
    lon_values = np.asarray(lon.values, dtype=float)

    # Signed cyclic difference in the range [-180, 180)
    lon_distance = np.abs(
        ((lon_values - value + 180.0) % 360.0) - 180.0
    )

    return int(np.argmin(lon_distance))


def load_subsetdata(
    lat,
    lon,
    lev,
    lat_of_interest,
    lon_of_interest,
    lev_of_interest,
):
    # Robust nearest-neighbor indices
    ilat_of_interest = nearest_coordinate_index(
        lat,
        lat_of_interest,
    )
    ilon_of_interest = nearest_longitude_index(
        lon,
        lon_of_interest,
    )
    ilev_of_interest = nearest_coordinate_index(
        lev,
        lev_of_interest,
    )

    ilat_of_interest_str = f"{ilat_of_interest:05d}"
    ilon_of_interest_str = f"{ilon_of_interest:05d}"
    ilev_of_interest_str = f"{ilev_of_interest:05d}"

    dir_o = "/glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/"

    search_pattern = (
        dir_o
        + "*/tensor_steps24_parrallel_lev"
        + ilev_of_interest_str
        + "_lat"
        + ilat_of_interest_str
        + "_lon"
        + ilon_of_interest_str
        + ".npy"
    )

    files = sorted(glob.glob(search_pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            "No IG files found for "
            f"lev index {ilev_of_interest}, "
            f"lat index {ilat_of_interest}, "
            f"lon index {ilon_of_interest}."
        )

    files_JJA = files[151:243]
    files_DJF = np.delete(
        np.asarray(files),
        np.s_[59:334],
        axis=0,
    )

    if len(files_JJA) == 0 or len(files_DJF) == 0:
        raise ValueError(
            f"Seasonal file selection is empty. "
            f"Found {len(files)} total files, "
            f"{len(files_JJA)} JJA files, and "
            f"{len(files_DJF)} DJF files."
        )

    IG_allvars_JJA = np.stack(
        [np.load(f).squeeze() for f in files_JJA]
    )
    IG_allvars_DJF = np.stack(
        [np.load(f).squeeze() for f in files_DJF]
    )

    return (
        IG_allvars_JJA,
        IG_allvars_DJF,
        ilat_of_interest,
        ilon_of_interest,
    )





# def load_subsetdata(lat, lon, lev, lat_of_interest, lon_of_interest, lev_of_interest):
#     # Find associated index value for getting data
#     lat_of_interest_near = lat.sel(lat=lat_of_interest, method='nearest')
#     lon_of_interest_near = lon.sel(lon=lon_of_interest, method='nearest')
#     lev_of_interest_near = lev.sel(lev=lev_of_interest, method='nearest')
    
#     ilat_of_interest = np.where(lat.values == lat_of_interest_near.values)[0][0]
#     ilon_of_interest = np.where(lon.values == lon_of_interest_near.values)[0][0]
#     ilev_of_interest = np.where(lev.values == lev_of_interest_near.values)[0][0]
    
#     ilat_of_interest_str = f"{ilat_of_interest:05}"
#     ilon_of_interest_str = f"{ilon_of_interest:05}"
#     ilev_of_interest_str = f"{ilev_of_interest:05}"

#     # grab all dates for the U level, lat, and lon point
#     dir_o = "/glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/"
#     files = sorted(
#         glob.glob(dir_o+"*/tensor_steps24_parrallel_lev"+ilev_of_interest_str+"_lat"+ilat_of_interest_str+"_lon"+ilon_of_interest_str+".npy")
#     )
#     files_JJA = files[151:243]
#     files_DJF = np.delete(files, np.s_[59:334], axis=0) #remove non-DJF
    
#     IG_allvars_JJA = np.stack([np.load(f).squeeze() for f in files_JJA])
#     IG_allvars_DJF = np.stack([np.load(f).squeeze() for f in files_DJF])

#     return IG_allvars_JJA, IG_allvars_DJF, ilat_of_interest, ilon_of_interest


def spatial_cube_subset(
    IG_allvars,
    addval,
    lat,
    lon,
    lat_range_of_interest,
    lon_range_of_interest,
    ilat_of_interest,
    ilon_of_interest,
):
    """
    Extract a latitude-longitude window.

    Latitude is clipped at the poles.
    Longitude wraps periodically.
    """

    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.asarray(lon.values, dtype=float)

    delta_lat = float(np.median(np.abs(np.diff(lat_values))))
    delta_lon = float(np.median(np.abs(np.diff(lon_values))))

    nlat_radius = int(
        np.rint(lat_range_of_interest / delta_lat)
    )
    nlon_radius = int(
        np.rint(lon_range_of_interest / delta_lon)
    )

    # Include both endpoints and the central point.
    lat_indices = np.arange(
        ilat_of_interest - nlat_radius,
        ilat_of_interest + nlat_radius + 1,
    )

    # Clip latitude instead of wrapping through negative indices.
    lat_indices = lat_indices[
        (lat_indices >= 0)
        & (lat_indices < lat_values.size)
    ]

    # Construct unwrapped indices first, then wrap them periodically.
    longitude_index_unwrapped = np.arange(
        ilon_of_interest - nlon_radius,
        ilon_of_interest + nlon_radius + 1,
    )

    lon_indices = longitude_index_unwrapped % lon_values.size

    # First select the requested variable and all spatial points.
    variable_subset = IG_allvars[
        :,
        addval:addval + 32,
        :,
        :,
    ]

    # np.take handles arbitrary index arrays cleanly.
    cube_subset = np.take(
        variable_subset,
        lat_indices,
        axis=-2,
    )
    cube_subset = np.take(
        cube_subset,
        lon_indices,
        axis=-1,
    )

    selected_lats = lat_values[lat_indices]
    selected_lons = lon_values[lon_indices]

    # Convert a wrapped sequence such as
    # [345, ..., 358.75, 0, 1.25, ..., 15]
    # into a continuous sequence such as
    # [-15, ..., 0, ..., 15].
    selected_lons = np.rad2deg(
        np.unwrap(
            np.deg2rad(selected_lons)
        )
    )

    # Shift the unwrapped coordinates so that the central longitude
    # matches its native coordinate value.
    center_position = nlon_radius
    center_native = lon_values[ilon_of_interest]
    center_unwrapped = selected_lons[center_position]

    selected_lons += 360.0 * np.round(
        (center_native - center_unwrapped) / 360.0
    )

    return cube_subset, selected_lats, selected_lons


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

def make_eof_ds(
    eof_reshape,
    pc,
    eigvals,
    lats,
    lons,
    levs,
    eof_nums,
):
    eof_nums = np.asarray(eof_nums)

    return xr.Dataset(
        {
            "eof_map": xr.DataArray(
                eof_reshape,
                dims=["lev", "lat", "lon", "eof"],
                coords={
                    "lev": levs,
                    "lat": lats,
                    "lon": lons,
                    "eof": eof_nums,
                },
            ),
            "pc": xr.DataArray(
                pc,
                dims=["time", "eof"],
                coords={
                    "time": np.arange(pc.shape[0]),
                    "eof": eof_nums,
                },
            ),
            "eigenvalues": xr.DataArray(
                eigvals,
                dims=["eof"],
                coords={"eof": eof_nums},
            ),
        }
    )