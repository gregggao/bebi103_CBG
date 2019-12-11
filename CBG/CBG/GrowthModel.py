import numpy as np
import pandas as pd

import multiprocessing
import warnings

import scipy.optimize
import scipy.stats as st
import scipy.special as sp

import bokeh
import bokeh_catplot
import bokeh.io
bokeh.io.output_notebook()

import bebi103

import holoviews as hv
hv.extension('bokeh')

def growth_event_id(df):
    # growth event identifier
    df = x
    frames = len(df['area (µm²)'])
    previous_area = 0
    growth_event_id = -1
    growth_event = []
    thes = 0.8
    # loop through frames
    for i in range(frames):
        current_area = df['area (µm²)'][i]
        diff = abs(current_area - previous_area)
        # check the area difference between frames
        if diff > thes:
            previous_area = current_area #update area
            growth_event_id = growth_event_id + 1 # reassign growth event
            growth_event.append(growth_event_id)

        else:
            previous_area = current_area #update area
            growth_event.append(growth_event_id)
    df['growth event'] = growth_event
    return df

def growth_time(df):
    # this is a function for ecdf calculation, returns the time of each growth event
    lens = len(df)
    growth_event = df["growth event"]
    time = df["time (min)"]
    time_diff = []
    previous_growth_event = 0
    for i in range(lens):
        current_time = time[i]
        current_growth_event = growth_event[i]
        diff = abs(current_growth_event - previous_growth_event)
        if diff == 1:
            time_diff.append(i)
            previous_growth_event = current_growth_event
        else:
            previous_growth_event = current_growth_event
    time_diff = time_diff[1:]
    time_diff = [time_diff[i+1]-time_diff[i] for i in range(len(time_diff)-1)]
    return time_diff

def ecdf_vals(df, legend_label = 'label', x_axis_label = 'x', y_axis_label = 'ECDF'):
    data = growth_time(df)
    #Create sorted data and ECDF value
    ecdf = np.array([np.sort(data), np.arange(1, len(data)+1)/len(data)])
    
    #Create data frame
    df = pd.DataFrame(np.transpose(ecdf), columns = [x_axis_label, y_axis_label])
    
    #Plot ECDF
    plot = hv.Points(
        data = df,
        kdims = [x_axis_label, y_axis_label],
        #Add label in order to obtain a legend on multiple plots
        label = legend_label
    )
    
    return plot, ecdf

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