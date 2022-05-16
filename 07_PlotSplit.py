#!/usr/bin/env/python3
'''
Description: A script for machine learning based alogorithm for splitting sample data into two sets randomly: training and testing plots.
the script splits plots in such a way that pixels are also allocated in ~70:30 proportions in the training and testing plots.
For any community chosen it creates 2 CSVs and 2 shape files, both corresponding to the training and testing set.

It reads the original shape file with the ID_1 column as well as CSV file created by the 06_HDFtoPandas_composites.py
Authors: Suvarna Punalekar (Aberystwyth University).
date 6 Sept 2021
'''

import glob, os, sys
from os import path
from sys import stdout
import numpy as np
from scipy import stats
import pandas as pd
import geopandas as gpd



'''
Input, Output directory and filenames
'''
#Input Shape file
gdf = gpd.read_file('TrainingData_ver3Plots_id.shp')
print(gdf.shape)
# Input with file with extracted composite values
CSVFile = pd.read_csv('SemiNatural_CompositeExtract_ver3Plots/SemiNatural_S1S2Terrain_ver3Plots.csv')
print(CSVFile.shape)

OutRoot = 'Ver3Plots_split_set2'
if not os.path.exists(OutRoot):
    os.mkdir(OutRoot)
    
##################################################################################
# Linking CSVfile with ID_1
DomSps = gdf.drop(columns='geometry')
#DomSps = DomSps[['ID_1', 'PRI_SPCODE', 'Quad_size']]

DomSps['ID_1'] = DomSps['ID_1'].astype(int)
print(DomSps.dtypes)

EOData = pd.merge(CSVFile, DomSps, how='inner', on=['ID_1'])
print(EOData.head())
##################################################################################
# Subset shapefile for species
TypeList = gdf['type'].unique()
print(TypeList)

# Select community/species to split polygon
TypeList = ['Vaccinium']
# 
for Type in TypeList:
    
    gdf_subset =gdf[gdf['type'] == Type]
    EOData_subset = EOData[EOData['type'] == Type]
    print('Shape of gdf and EOData for ' + Type)
    print(gdf_subset.shape, EOData_subset.shape)


    ##################################################################################
    # Splitting the subset
    num1 = 0
    num2 = 10000   # Set high values, it is just number of loops to run. The loops keeps running until this number is hit or it breaks if a good split is obtained.
    frac_poly = 0.80  # this number is a fraction. Play with this if good split is not obtained with 0.80. It defines how many polygons to choose for a training set.
    while num1 < num2:
        gdf_subset_cal = gdf_subset.sample(frac=frac_poly, replace=False)
        ListID_cal = gdf_subset_cal.ID_1.unique()
        #print(ListID_cal)
        EOData_subset_cal = EOData_subset[EOData_subset['ID_1'].isin(ListID_cal)]
        if (EOData_subset_cal.shape[0] >= 0.69*EOData_subset.shape[0]) & (EOData_subset_cal.shape[0] < 0.71*EOData_subset.shape[0]):
            print('Good split')
            print('Shape of gdf_subset_cal and EOData_subset_cal for ' +  Type)
            print(gdf_subset_cal.shape, EOData_subset_cal.shape)
            
            gdf_subset_val = gdf_subset.drop(gdf_subset_cal.index)
            EOData_subset_val = EOData_subset.drop(EOData_subset_cal.index)
            print('Shape of gdf_subset_val and EOData_subset_val for ' + Type)
            print(gdf_subset_val.shape, EOData_subset_val.shape)
            
            gdf_subset_cal.to_file(OutRoot + '/' + Type + '_calsubset.shp')
            gdf_subset_val.to_file(OutRoot + '/' + Type + '_valsubset.shp')
            EOData_subset_cal.to_csv(OutRoot + '/' + Type + '_calsubset.csv', sep=',', index=False)
            EOData_subset_val.to_csv(OutRoot + '/' + Type + '_valsubset.csv', sep=',', index=False)
            break
        num1 = num1 +1
        print('Not a good split for ' + str(num1))
        print('Shape of gdf_subset_cal and EOData_subset_cal')
        print(gdf_subset_cal.shape, EOData_subset_cal.shape)
        del gdf_subset_cal, EOData_subset_cal

##################################################################################
