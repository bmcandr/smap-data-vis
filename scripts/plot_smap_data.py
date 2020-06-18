'''
This script reads SMAP data and plots it.
'''

# import libraries/modules needed:

# the os module is used to create paths to data
import os

# the Path module is used to get paths to files of interest
from pathlib import Path as path

# matplotlib is used to plot data
# (anim is a submodule that can be used to create animations)
from matplotlib import pyplot as plt
# from matplotlib import animation as anim

# Basemap is a module used for plotting maps
from mpl_toolkits.basemap import Basemap

# rasterio is a module used to read/write geospatial raster data
import rasterio as rio

# rasterio reads raster data into numpy arrays so we need numpy too
import numpy as np

# creating paths using the os.path.join method enables
# cross-platform compatability
DATA_DIR = os.path.join('..', 'data')
INPUT_DATA_DIR = os.path.join(DATA_DIR, 'input')
SMAP_DATA_DIR = os.path.join(INPUT_DATA_DIR, 'SMAP', 'SPL3SMP.005')

# file suffix pattern for the SMAP files of interest
SMAP_FILE_SUFFIX = '*_R16010_001.h5'

# get paths to all HDF5 files ending with SMAP_FILE_SUFFIX
# found in all directories within SMAP_DATA_DIR
DATA_PATHS = sorted([str(filepath) for filepath in path(SMAP_DATA_DIR).rglob(SMAP_FILE_SUFFIX)])

# get the first in path DATA_PATHS to use for testing
TEST_DATA_PATH = DATA_PATHS[0]

# the HDF5 files contain many datasets, but
# the soil_moisture dataset is what we're interested in
SM_DATASET_NAME = 'soil_moisture'

# create a path to an output directory in case
# we want to save files at some point
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# check if OUTPUT_DIR exists and, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# create variables to hold path to files containing EASE grid lat/lons (used for plotting)
# data source: https://github.com/TUW-GEO/ease_grid/tree/master/tests/test_data
EASE_GRID_DIR = os.path.join(DATA_DIR, 'ease_grid')
EASE_LATS_PATH = os.path.join(EASE_GRID_DIR, 'EASE2_M36KM.lats.964x406x1.double')
EASE_LONS_PATH = os.path.join(EASE_GRID_DIR, 'EASE2_M36KM.lons.964x406x1.double')

# read EASE2 lat/lons and reshape from a 1D array into a 2D array matching the EASE Grid shape
EASE_SHAPE = (406, 964)
EASE_LATS = np.fromfile(EASE_LATS_PATH, dtype=np.float64).reshape(EASE_SHAPE)
EASE_LONS = np.fromfile(EASE_LONS_PATH, dtype=np.float64).reshape(EASE_SHAPE)

# store the EPSG projection code for the EASE Grid
# that this data is projected in
# this info is useful if we want to reproject the data at some point
EASE_EPSG = 'epsg:3410'

# set output figure size
FIG_SIZE = (30, 10)

def print_dataset_names(src_path: str):
    '''
    Print name of datasets in file at src_path.
    '''

    with rio.open(src_path) as src:
        subdatasets = src.subdatasets
        for name in subdatasets:
            print(name)


def read_dataset(src_path: str, dataset_name: str, mask_nodata: bool = True):
    '''
    Read dataset_name in HDF5 file at src_path
    '''

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
    '''
    Read and return lat/lon from HDF5 file at src_path.
    '''

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
    '''
    Save plots of data.
    '''

    for data_path in data_paths:
        plot_date = data_path.split('/')[-2]
        data, meta, desc = read_dataset(data_path, dataset_name)
        plot_data(data, dataset_name, desc)
        plt.title(plot_date)

        out_fp = os.path.join(OUTPUT_DIR, f'{plot_date}.png')
        plt.savefig(out_fp)


def calc_mean_dataset(data_paths: list, dataset_name: str):
    '''
    Read in data and calculate mean.
    '''

    datasets = []
    meta = None
    desc = None
    for data_path in data_paths:
        dataset, meta, desc = read_dataset(data_path, dataset_name)
        datasets.append(dataset)

    stacked_datasets = np.dstack(datasets)

    mean_data = np.nanmean(stacked_datasets, axis=2)

    return mean_data, meta, desc


def plot_data(data: np.ndarray, \
                    dataset_name: str, dataset_desc: dict, \
                    lats: np.ndarray = EASE_LATS, lons: np.ndarray = EASE_LONS, \
                    projection: str = 'cyl', cmap: str = 'viridis_r'):

    '''
    Plot data.
    '''

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

    fig, ax = plt.subplots(figsize=FIG_SIZE)

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

    plt.show()


def test_plot_mean():
    '''
    Test plotting mean of all soil moisture datasets
    '''

    mean_sm, meta, desc = calc_mean_dataset(DATA_PATHS, SM_DATASET_NAME)
    plot_data(mean_sm, SM_DATASET_NAME, desc)

test_plot_mean()
