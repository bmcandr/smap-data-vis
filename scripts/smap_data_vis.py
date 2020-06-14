"""
This script reads SMAP data and plots it.
"""

import os

from pathlib import Path as path
from matplotlib import pyplot as plt
# from matplotlib import animation as anim
from mpl_toolkits.basemap import Basemap

import rasterio as rio
import numpy as np

DATA_DIR = os.path.join('data', 'testcase5_ldt_obscdf')
RS_DATA_DIR = os.path.join(DATA_DIR, 'RS_DATA')
SMAP_DATA_DIR = os.path.join(RS_DATA_DIR, 'SMAP', 'SPL3SMP.005')
SM_FILE_SUFFIX = '*_R16010_001.h5'

EASE_GRID_DIR = os.path.join('data', 'ease_grid')
EASE_LATS_PATH = os.path.join(EASE_GRID_DIR, 'EASE2_M36KM.lats.964x406x1.double')
EASE_LONS_PATH = os.path.join(EASE_GRID_DIR, 'EASE2_M36KM.lons.964x406x1.double')

PLOT_OUT_DIR = os.path.join('data', 'output', 'sm_anim_frames')

SM_DATASET_NAME = 'soil_moisture'

TEST_DATA_PATH = os.path.join(SMAP_DATA_DIR, '2017.03.01', SM_FILE_SUFFIX)

# get paths to all h5 files
DATA_PATHS = sorted([str(fp) for fp in path(SMAP_DATA_DIR).rglob(SM_FILE_SUFFIX)])

EASE_EPSG = 'epsg:3410'
WGS84_EPSG = 'epsg:3426'

# read EASE2 lat/lons
EASE_LATS = np.fromfile(EASE_LATS_PATH, dtype=np.float64).reshape((406, 964))
EASE_LONS = np.fromfile(EASE_LONS_PATH, dtype=np.float64).reshape((406, 964))

def print_dataset_names(src_path: str):
    """
    Print name of datasets in file at src_path.
    """

    with rio.open(src_path) as src:
        subdatasets = src.subdatasets
        for name in subdatasets:
            print(name)


def read_dataset(src_path: str, dataset_name: str, mask_nodata: bool = True):
    """
    Read dataset_name in HDF5 file at src_path
    """

    with rio.open(src_path) as src:
        subdatasets = src.subdatasets
        data = ''
        for data_path in subdatasets:
            name = data_path.split('/')[-1]
            if name == dataset_name:
                with rio.open(data_path) as subdataset:
                    data = subdataset.read(1)
                    meta = subdataset.meta
                    desc = subdataset.tags(bidx=1) # additional metadata (e.g., units)
                    meta['crs'] = EASE_EPSG

                    if mask_nodata:
                        data[data == meta['nodata']] = np.nan
    return data, meta, desc


def read_coords(src_path: str):
    """
    Read and return lat/lon from HDF5 file at src_path.
    """

    with rio.open(src_path) as src:
        subdatasets = src.subdatasets
        lat = ''
        lon = ''
        for data_path in subdatasets:
            name = data_path.split('/')[-1]
            if name == 'longitude':
                with rio.open(data_path) as subdataset:
                    lon = subdataset.read(1)
            elif name == 'latitude':
                with rio.open(data_path) as subdataset:
                    lat = subdataset.read(1)
    return lat, lon


def save_plots(data_paths: list, dataset_name: str):
    """
    Save plots of data.
    """

    for data_path in data_paths:
        plot_date = data_path.split('/')[-2]
        data, meta, desc = read_dataset(data_path, dataset_name)
        plot_data(data, dataset_name, desc)
        plt.title(plot_date)

        out_fp = os.path.join(PLOT_OUT_DIR, f'{plot_date}.png')
        plt.savefig(out_fp)


def plot_data(data: np.ndarray, \
                    dataset_name: str, dataset_desc: dict, \
                    lats: np.ndarray = EASE_LATS, lons: np.ndarray = EASE_LONS, \
                    projection: str = 'cyl', cmap: str = 'viridis_r'):

    """
    Plot data.
    """

    long_name, units, vmin, vmax = (None, None, None, None)

    for descriptor in dataset_desc.keys():
        if 'long_name' in descriptor:
            long_name = dataset_desc[descriptor]
        elif 'units' in descriptor:
            units = dataset_desc[descriptor]
        elif 'valid_min' in descriptor:
            vmin = dataset_desc[descriptor]
        elif 'valid_max' in descriptor:
            vmax = dataset_desc[descriptor]

    fig, ax = plt.subplots(figsize=(30, 10))

    if projection == 'robin':
        map_plot = Basemap(projection='robin', resolution='c', lat_0=0, lon_0=0)
        lons, lats = map_plot(lons, lats)
    else:
        map_plot = Basemap(projection='cyl', resolution='l', ax=ax)

    ax.set_title(long_name.split('.')[0].upper(), pad=40)

    cs = map_plot.pcolor(lons, lats, data, latlon=True, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = map_plot.colorbar(mappable=cs, location='bottom', pad='5%')
    cbar.set_label(f'{dataset_name.upper()} ({units})')

    map_plot.drawcoastlines()
    map_plot.drawcountries()

    paralells = np.arange(-90, 91, 30)
    meridians = np.arange(-180, 180, 30)
    labels = [True for i in paralells]

    map_plot.drawparallels(paralells, color='gray', labels=labels)
    map_plot.drawmeridians(meridians, color='gray', labels=labels)
    map_plot.drawlsmask(ocean_color='lightblue')

    # plt.tight_layout()
    plt.show()

def gen_mean_dataset(data_paths: list, dataset_name: str):
    """
    Read in data and calculate mean.
    """

    datasets = []
    meta = None
    desc = None
    for data_path in data_paths:
        dataset, meta, desc = read_dataset(data_path, dataset_name)
        datasets.append(dataset)

    return np.dstack(datasets), meta, desc


def test_plot_mean():
    """
    Test plotting mean of all soil moisture datasets
    """

    test_dataset, test_meta, test_desc = gen_mean_dataset(DATA_PATHS, SM_DATASET_NAME)
    mean_sm = np.nanmean(test_dataset, axis=2)
    plot_data(mean_sm, SM_DATASET_NAME, test_desc)

test_plot_mean()
