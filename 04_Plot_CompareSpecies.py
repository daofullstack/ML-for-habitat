##############################################################################
'''
Script to plot profiles using output of 03_VIValues_groupby_months.py
Author : Suvarna M. Punalekar
Date : 6 Sept 20221
'''
##############################################################################

import glob, os, sys
from os import path
import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import geopandas as gpd

bandn = 'VARIg'

title = ['2015to2019', '2018', '2019', '2020']
Files1 = ['Plot890_VI_plotmonthlymedian_2015to2019.csv']
#Files1 = ['Plot890_VI_plotmonthlymedian_2015to2019.csv', 'Plot890_VI_plotmonthlymedian_2018.csv',  'Plot890_VI_plotmonthlymedian_2019.csv', 'Plot890_VI_plotmonthlymedian_2020.csv']
#Files2 = ['WyeV_VI_plotmonthlystdev_2015to2019.csv', 'WyeV_VI_plotmonthlystdev_2018.csv',  'WyeV_VI_plotmonthlystdev_2019.csv']

Indir = 'SemiNatural_BandExtract_Plot890/'

fig1, axes = plt.subplots(nrows=2, ncols=4)


sps1 = 'Grass_other'
sps2 = 'WetHeath'
sps3 = 'Molinia'

j = 0
k = 0
for plotn in range(0,4):
    f1 = Files1[plotn]

    
    df = pd.read_csv(Indir + f1, sep=',')
    df1 = df[df['type'] == sps1]
    df2 = df[df['type'] == sps2]
    df3 = df[df['type'] == sps3]
    
    month_med_sps1 = df1.groupby(['Month'])[bandn].median().reset_index()
    month_std_sps1 = df1.groupby(['Month'])[bandn].std().reset_index()
    
    month_med_sps2 = df2.groupby(['Month'])[bandn].median().reset_index()
    month_std_sps2 = df2.groupby(['Month'])[bandn].std().reset_index()
    
    month_med_sps3 = df3.groupby(['Month'])[bandn].median().reset_index()
    month_std_sps3 = df3.groupby(['Month'])[bandn].std().reset_index()
    
    if plotn == 0:
        ymax = df1[bandn].max()
        ymin = df1[bandn].min()
        ymin = ymin - 0.01*abs(ymin)
        #ymax = 250
        ymax = ymax + 0.01*abs(ymax)
        #ymin =715
    ##############################################################################

    sps1_id = df1.ID_1.unique()
    #print(sps1_id)
    for i in range(0,len(sps1_id)):
        idn = sps1_id[i]
        sps1_subset = df1[df1['ID_1'] ==idn]
        sps1_subset.plot(kind='line', x='Month', y=bandn, color='red', ax=axes[j,k], title= title[plotn], legend=False)
        axes[j,k].set_xlim(0,13)
    
    sps2_id = df2.ID_1.unique()
    #print(sps2_id)
    for i in range(0,len(sps2_id)):
        idn = sps2_id[i]
        sps2_subset = df2[df2['ID_1'] ==idn]
        sps2_subset.plot(kind='line', x='Month', y=bandn, color='blue', ax=axes[j,k], legend=False)
        axes[j,k].set_xlim(0, 13)
        axes[j,k].set_ylim(ymin, ymax)
        axes[j,k].set_ylabel(bandn)
       
    # sps3_id = df3.ID_1.unique()
    # #print(sps3_id)
    # for i in range(0,len(sps3_id)):
    #     idn = sps3_id[i]
    #     sps3_subset = df3[df3['ID_1'] ==idn]
    #     sps3_subset.plot(kind='line', x='Month', y=bandn, color='green', ax=axes[j,k], legend=False)
    #     axes[j,k].set_xlim(0, 13)
    #     axes[j,k].set_ylim(ymin, ymax)
    #     axes[j,k].set_ylabel(bandn)   
        
    ##############################################################################
    
    err = month_std_sps1[bandn].values
    month_med_sps1.plot(kind='line', x='Month', y=bandn, color='red', yerr = err, ax=axes[j+1,k], label=sps1, title = title[plotn])
    
    err = month_std_sps2[bandn].values
    month_med_sps2.plot(kind='line', x='Month', y=bandn, color='blue', yerr = err, ax=axes[j+1,k], label=sps2)
        
    # err = month_std_sps3[bandn].values
    # month_med_sps3.plot(kind='line', x='Month', y=bandn, color='green', yerr = err, ax=axes[j+1,k], label=sps3)
    
    axes[j+1,k].set_xlabel('Months')
    axes[j+1,k].set_ylabel(bandn)
    axes[j+1,k].set_ylim(ymin, ymax)
    axes[j+1,k].set_xlim(0, 13)
    
    k = k+1
    
    del df1, df2, month_med_sps1, month_med_sps2, month_med_sps3, month_std_sps1, month_std_sps2, month_std_sps3, err


plt.show()
