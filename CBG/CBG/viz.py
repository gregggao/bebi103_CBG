import multiprocessing
import warnings

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
        data=(t_theor, Expon_fun(a0, k, t_theor)),
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


def Expon_all_params (df):
    # number of total growth events
    num_growth = df['growth event'].values[-1] +1
    params_list = []
    for i in range(num_growth):
        # for each growth event, the mle is calculated
        df_one_growth = df.loc[df['growth event'] == i]
        # set the threshold for complete recognition of growth event
        if len(df_one_growth) > 75:
            t = df_one_growth['time (min)'].values
            growth = df_one_growth['area (µm²)'].values
            if i != 0:
                t = [x - t[0] + 1  for x in t]
                t = np.asarray(t)
            params = mle_Expon_fun(t, growth)
            # collect mles in a list
            params_list.append(params)
        else:
            # give the fragments no parameters to drop them
            params_list.append([0,0,0])
    
    params_list = pd.DataFrame(data=np.vstack(params_list), columns=['a0', 'k', 'sigma'])
    
    return params_list

def Expon_fun(a0, k, t):
    # Exponential mathematical model
    a = a0*np.exp(k*t)
    return a

def Expon_log_likelihood(params, t, growth):
    # there are 3 params, 4 phyiscals and sigma from generative function
    a0, k, sigma = params
    # restrain params
    if a0 < 0 or k < 0 or k > 1 or sigma < 0:
         return -np.inf
    # mu is the expected value
    mu = Expon_fun(a0, k, t)
    return np.sum(st.norm.logpdf(growth, mu, sigma))

def mle_Expon_fun(t, growth):
    # params optimized by minimizing negated log likelihood 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t, growth: -Expon_log_likelihood(params, t, growth),
            x0=np.array([1.4, 0.006, 0.05]),
            args=(t, growth),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def Linear_all_params (df):
    # number of total growth events
    num_growth = df['growth event'].values[-1] +1
    params_list = []
    for i in range(num_growth):
        # for each growth event, the mle is calculated
        df_one_growth = df.loc[df['growth event'] == i]
        # set the threshold for complete recognition of growth event
        if len(df_one_growth) > 75:
            t = df_one_growth['time (min)'].values
            growth = df_one_growth['area (µm²)'].values
            if i != 0:
                t = [x - t[0] + 1  for x in t]
                t = np.asarray(t)
            params = mle_Linear_fun(t, growth)
            # collect mles in a list
            params_list.append(params)
        else:
            # give the fragments no parameters to drop them
            params_list.append([0,0,0])
            
    params_list = pd.DataFrame(data=np.vstack(params_list), columns=['a0', 'k', 'sigma'])

    return params_list

def Linear_fun(a0, k, t):
    # Linear mathematical model
    a = a0 + a0*k*t
    return a

def Linear_log_likelihood(params, t, growth):
    # there are 3 params, 4 phyiscals and sigma from generative function
    a0, k, sigma = params
    # restrain params
    if a0 < 0 or k < 0 or k > 1 or sigma < 0:
         return -np.inf
    # mu is the expected value
    mu = Linear_fun(a0, k, t)
    return np.sum(st.norm.logpdf(growth, mu, sigma))

def mle_Linear_fun(t, growth):
    # params optimized by minimizing negated log likelihood 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t, growth: -Linear_log_likelihood(params, t, growth),
            x0=np.array([1.4, 0.008, 0.05]),
            args=(t, growth),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)