import numpy as np
import pandas as pd
import panel as pn
import glob
import os

import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.io
import skimage.morphology
import skimage.segmentation

import bokeh
import holoviews as hv
hv.extension('bokeh')

import bebi103
bebi103.hv.set_defaults()

def tiF_Loader(tif):
    # Glob string for images
    im_glob = os.path.join(tif)
    # Get list of files in directory
    im_list = sorted(glob.glob(im_glob))
    #read in the stacks
    im = skimage.io.imread(im_list[0])
    return im

def image_process(im):
    ip = 0.052
    # create empty list to hold the data for each image in the stacks
    stack_data = []

    # same threshold as defined in 5.1
    thresh = 4300

    # loops to process the images
    for image in im:
        im_bw = image < thresh  # thresholding
        im_bw = skimage.morphology.remove_small_objects(
            im_bw, min_size=100
        )  # remove small objects
        im_bw = skimage.segmentation.clear_border(
            im_bw
        )  # here we get rid of the cell in the corner
        im_data = np.sum(np.sum(im_bw))  # get the area of the cell by boolean indexing
        if im_data != 0:  # don't append any empty frames
            im_data = im_data*ip*ip
            stack_data.append(im_data)
    
    # create data frames containing the area
    df = pd.DataFrame(data=stack_data, columns=["area (µm²)"])
    return df

def im_thresholded(im):
    ip = 0.052
    # create empty lists to hold the thresholded image stacks
    im_thresholded = []

    # same threshold as defined in 5.1
    thresh = 4300

    # loops to process the images
    for image in im:
        im_bw = image < thresh  # thresholding
        im_bw = skimage.morphology.remove_small_objects(
            im_bw, min_size=100
        )  # remove small objects
        im_bw = skimage.segmentation.clear_border(
            im_bw
        )  # here we get rid of the cell in the corner
        im_data = np.sum(np.sum(im_bw))  # get the area of the cell by boolean indexing
        if im_data != 0:  # don't append any empty frames
            im_data = im_data*ip*ip
            im_thresholded.append(im_bw)
    return im_thresholded