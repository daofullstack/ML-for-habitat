##############################################################################
'''
Script to prepare inputs for the Machine Learning algorithm.
The oututs are two CSV files
one corresponding to the Training data and the other Testing data.
The script merges all the CSV files created in the either of the training or tesing set as produced by 07_PlotSplit.py
Author : Suvarna M. Punalekar
Date : 6 Sept 2021
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

##############################################################################
# Read CSV File with Individual Bands
InDir = 'Ver3Plots_split_set2/'
# 
CalCSVList = ['Juncus_marshy_calsubset.csv', 'WetHeath_calsubset.csv', 'Molinia_calsubset.csv', 'Swamp_calsubset.csv',
           'Vaccinium_calsubset.csv', 'Grassland_calsubset.csv', 'Calluna_calsubset.csv',
           'Bracken_calsubset.csv', 'Ulex_calsubset.csv']

# CalCSVList = ['Swamp_calsubset.csv','Calluna_calsubset.csv',
#            'Bracken_calsubset.csv', 'Ulex_calsubset.csv', 'Juncus_marshy_calsubset.csv']

ValCSVList = ['Juncus_marshy_valsubset.csv', 'WetHeath_valsubset.csv', 'Molinia_valsubset.csv', 'Swamp_valsubset.csv',
           'Vaccinium_valsubset.csv', 'Grassland_valsubset.csv', 'Calluna_valsubset.csv',
            'Bracken_valsubset.csv', 'Ulex_valsubset.csv']
# ValCSVList = [ 'Swamp_valsubset.csv','Calluna_valsubset.csv',
#            'Bracken_valsubset.csv', 'Ulex_valsubset.csv', 'Juncus_marshy_valsubset.csv']

df = pd.read_csv(InDir + 'Calluna_calsubset.csv', sep=',')
columns = list(df.columns)
print(columns)
del df

#######################################################################################################################
dfCal = pd.DataFrame(columns = columns)

for CSV in CalCSVList:
    df = pd.read_csv(InDir + CSV, sep=',')
    dfCal = dfCal.append(df, ignore_index = True)
    
    del df
dfCal = dfCal.drop(columns = ['id', 'plotsize', 'source', 'layer', 'path', 'Unnamed: 0']).sort_values('typecode', ascending = True)
print(dfCal.head())
print(dfCal.shape)

dfCal.to_csv(InDir + 'AllSps_CalSet.csv', sep=',', index = False)

del dfCal
#######################################################################################################################
dfVal = pd.DataFrame(columns = columns)

for CSV in ValCSVList:
    df = pd.read_csv(InDir + CSV, sep=',')
    dfVal = dfVal.append(df, ignore_index = True)
    
    del df
dfVal = dfVal.drop(columns = ['id', 'plotsize', 'source', 'layer', 'path', 'Unnamed: 0']).sort_values('typecode', ascending = True)
print(dfVal.head())
print(dfVal.shape)
dfVal.to_csv(InDir + 'AllSps_ValSet.csv', sep=',', index = False)
del dfVal