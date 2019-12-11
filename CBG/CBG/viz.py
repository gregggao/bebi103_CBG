import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special as sp

import bokeh_catplot
import bebi103

import bokeh.io
bokeh.io.output_notebook()

import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()

def data_plotter(df):
    # plot the data
    return hv.Scatter(
        data=df,
        kdims=['time (min)','area (µm²)'],
    ).opts(
        height = 500,
        width =500,
        padding = 0.05,
        cmap = "magma",
        alpha = 0.2
    )

def plot_theor_Linear_k(a0, k):
    # set a theoretical x-axis data 0 to 100
    t_theor = np.linspace(0, 100, 1000)
    # plot the data
    return hv.Curve(
        data=(t_theor, Linear_fun(a0, k, t_theor)),
        kdims=['t_theor'],
        vdims=['Linear_fun'],
        label=f'a0, k = {round(a0,4),round(k,4)}',
    )

def plot_theor_Expon_k(a0, k):
    # set a theoretical x-axis data 0 to 100
    t_theor = np.linspace(0, 100, 1000)
    return hv.Curve(
        data=(t_theor, Expon_fun(1.3, k, t_theor)),
        kdims=['t_theor'],
        vdims=['Expon_fun'],
        label=f'a0, k = {round(a0,4),round(k,4)}',
    )

def Linear_plot_growth(df):
    # split the data to x and y axis
    params_list = Linear_all_params(df)
    # plot the data
    plots = data_plotter(df)
    # iterate through growth event
    for num_growth in range(len(params_list)):
        a0 = params_list['a0'][num_growth]
        k = params_list['k'][num_growth]
        # find the real time values corresponding to the theoretical curve
        ini_t = df.loc[df['growth event'] == num_growth]['time (min)'].values[0]
        final_t = df.loc[df['growth event'] == num_growth]['time (min)'].values[-1]
        # set the real time x-axis
        tx = np.linspace(ini_t, final_t, 1000)
        # plot ONE growth event
        t_theor = np.linspace(0, 100, 1000)
        plot = hv.Curve(
                data=(tx, Linear_fun(a0, k, t_theor)),
                kdims=['t_theor'],
                vdims=['Expon_fun'],
            )
        #plot the curve over the data
        plots = plots*plot
        
    return plots

def Expon_plot_growth(df):
    # split the data to x and y axis
    params_list = Expon_all_params(df)
    # plot the data
    plots = data_plotter(df)
    # iterate through growth event
    for num_growth in range(len(params_list)):
        a0 = params_list['a0'][num_growth]
        k = params_list['k'][num_growth]
        # find the real time values corresponding to the theoretical curve
        ini_t = df.loc[df['growth event'] == num_growth]['time (min)'].values[0]
        final_t = df.loc[df['growth event'] == num_growth]['time (min)'].values[-1]
        # set the real time x-axis
        tx = np.linspace(ini_t, final_t, 1000)
        # plot ONE growth event
        t_theor = np.linspace(0, 100, 1000)
        plot = hv.Curve(
                data=(tx, Expon_fun(a0, k, t_theor)),
                kdims=['t_theor'],
                vdims=['Expon_fun'],
            )
        #plot the curve over the data
        plots = plots*plot
    
    return plots