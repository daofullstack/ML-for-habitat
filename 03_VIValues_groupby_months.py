##############################################################################
'''
Script to calculate Vegetation indices using the extracted band values in the CSV file created after running 01_ExtractBandValues.py and 02_HDFtoPandas.py.
It also links the CSV file with the original shape file using ID_1 column so every pixel is associated with the polygon it was associated with, and hence class it belong too.
Multiyear medians as well as yearly medians are calculated and stored in separate CSV files.
These can be used for profile plotting.

Author : Suvarna M. Punalekar
Date : 6 Sept 2021
'''
##############################################################################

import glob, os, sys
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
# Apply filter to remove EO dates where number of pixels within the buffer were less than 4
InDir = 'Burnscars_BandExtract/'

OutFile_med = InDir + 'Plot890_VI_plotmonthlymedian_2015to2019.csv'
OutFile_std = InDir + 'Plot890_VI_plotmonthlystdev_2015to2019.csv'
OutFile2019_med = InDir + 'Plot890_VI_plotmonthlymedian_2019.csv'
OutFile2019_std = InDir + 'Plot890_VI_plotmonthlystdev_2019.csv'
OutFile2018_med = InDir + 'Plot890_VI_plotmonthlymedian_2018.csv'
OutFile2018_std = InDir + 'Plot890_VI_plotmonthlystdev_2018.csv'
OutFile2020_med = InDir + 'Plot890_VI_plotmonthlymedian_2020.csv'
OutFile2020_std = InDir + 'Plot890_VI_plotmonthlystdev_2020.csv'

CSVFile = pd.read_csv(InDir + 'Burnscars_BandExtract_UVC_2015to2021.csv', ',')

EOData = CSVFile.groupby(['ID_1', 'Date']).filter(lambda x: len(x) >= 4)
# Covert Int column to date format
EOData['DATE'] = pd.to_datetime(EOData['Date'].astype(str))
EOData = EOData.drop(columns='Date')
EOData['ID_1'] = EOData['ID_1'].astype(int)

DomSps = gpd.read_file('BurnScars.shp')
DomSps = DomSps.drop(columns='geometry')
print(DomSps.head)
#EOData = EOData['ID_1'].astype(int)
DomSps['ID_1'] = DomSps['ID_1'].astype(int)
print(DomSps.dtypes)

EOData = pd.merge(EOData, DomSps, how='inner', on=['ID_1'])

#print(EOData.dtypes)

EOData['Month'] = pd.DatetimeIndex(EOData['DATE']).month
EOData['Year'] = pd.DatetimeIndex(EOData['DATE']).year

###############################################################################
print(EOData.head())
print(EOData.tail())
band = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'NDVI', 'GNDVI', 'CLg',  'WDRVI',
        'ARVI', 'VARIg', 'CLre', 'PSRI', 'ARI', 'MCARI2', 'IRECI', 'S2REP' , 'REPO', 'NDMI', 'NBR', 'MSI']
#Calculate Indices
# 
# Conventional NIR indices, (Korhonen et al. (2017))
#B7 is replaced with B6 (rededge instead of NIR in CLg, WDRVI
EOData['CLg'] = (EOData['B6']/EOData['B2']) - 1  #Gitelson et al 2003
#
#old formula as per Korhonen et al.
#EOData['WDRVI'] = ((0.02 * EOData['B6'] - EOData['B3'])/(0.02 * EOData['B6'] + EOData['B3'])) + (0.98/1.02)
# formula as per Peng and Gitelson 2011.
EOData['WDRVI'] = ((0.2 * EOData['B6'] - EOData['B3'])/(0.2 * EOData['B6'] + EOData['B3'])) + (0.98/1.2)

# 
EOData['NDVI'] = (EOData['B7'] - EOData['B3']) / (EOData['B7'] + EOData['B3'])
# 
# ##############################################################################
# # Atmospheric indices, (Korhonen et al. (2017), Huete 2002, Kaufman 1992)
# in the original formula there is coeffint gamma, but
#Gitelson et al 1996 used gama = 1. Hence the formula below
EOData['ARVI'] = (EOData['B6'] - EOData['B3'] - (EOData['B1'] - EOData['B3'])) / (EOData['B6'] + EOData['B3'] - (EOData['B1'] - EOData['B3']))
# 
# EOData['EVI'] = 2.5 * (EOData['B6'] - EOData['B3']) / (EOData['B6'] + 6*EOData['B3'] - 7.5*EOData['B1'] + 1)
# 
EOData['VARIg'] = (EOData['B2'] - EOData['B3']) / (EOData['B2'] + EOData['B3'] - EOData['B1']) #Gitelson 2002
# 
# ##############################################################################
# # Red edge indices, (Korhonen et al. (2017))
# 
EOData['CLre'] = (EOData['B6']/EOData['B4']) - 1  #Peng and Gitelson 2011
# 
# EOData['WDRVIre'] = ((0.01 * EOData['B6'] - EOData['B4'])/(0.01 * EOData['B6'] + EOData['B4'])) + (0.99/1.01)
# 
EOData['PSRI'] = ((EOData['B3'] - EOData['B1'])) / EOData['B5'] #note that the index is changed as per S2 hub and also other papers. It is B1 rather than B2 as before
# 
# EOData['MTCI'] = ((EOData['B5'] - EOData['B4'])) / ((EOData['B4'] - EOData['B3']))
# 
EOData['MCARI2'] = 1.5*(2.5*(EOData['B6'] - EOData['B3']) - 1.3*(EOData['B6'] - EOData['B2']))/((2*EOData['B6'] + 1)**2 -(6*EOData['B6'] - 5*EOData['B3']**0.5) - 0.5)**0.5
# 
# EOData['TCARI'] = 3*((EOData['B4'] - EOData['B3']) - 0.2*(EOData['B4'] - EOData['B2'])*(EOData['B4']/EOData['B3']))
# 
# EOData['TCARI2'] = 3*((EOData['B5'] - EOData['B4']) - 0.2*(EOData['B5'] - EOData['B2'])*(EOData['B5']/EOData['B4']))
# 
# EOData['TVI'] = 0.5*(120*(EOData['B5'] - EOData['B2']) - 200*(EOData['B3'] - EOData['B2']))
# 
# EOData['MTVI2'] = 1.5*(1.2*(EOData['B6'] - EOData['B2']) - 2.5*(EOData['B3'] - EOData['B2'])) / ((2*EOData['B6'] + 1)**2 -(6*EOData['B6']- 5*EOData['B3']**0.5) - 0.5)**0.5
# 
EOData['IRECI'] = (EOData['B6'] - EOData['B3']) / (EOData['B4']/EOData['B5'])
# 
EOData['S2REP'] = 705 + 35*(((EOData['B6'] + EOData['B3'])/2) - EOData['B4'])/(EOData['B5'] - EOData['B4'])
# 
# EOData['MSRren'] = ((EOData['B8']/EOData['B4'])-1)/((EOData['B8']/EOData['B4']) + 1)**0.5
# 
# ##############################################################################
# #SWIR indices, (Korhonen et al. (2017))
# 
EOData['NDMI'] = (EOData['B8'] - EOData['B9']) / (EOData['B8'] + EOData['B9'])
# #similar to NDMI on Sentinel 2 hub, but called NDII in Korhonen paper
# 
EOData['NBR']  = (EOData['B8'] - EOData['B10']) / (EOData['B8'] + EOData['B10'])
# 
# EOData['NMDI'] = (EOData['B8'] - (EOData['B9']-EOData['B10'])) / (EOData['B8'] + (EOData['B9']-EOData['B10']))
# 
# ##############################################################################
# # Additional Indices from Sentinel 2 hub
# EOData['CHLrededge'] = (EOData['B6'] / EOData['B4'])**(-1)
# 
EOData['REPO'] = 700 + 40*((((EOData['B3']+EOData['B6'])/2)-EOData['B4'])/(EOData['B5']-EOData['B4'])) #Red edge position index (REPO), 700+40*((670nm+780nm/2)-700nm/(740nm-700nm),
# #The script is based on Gholizadeh et al., 2016. , min = 690, max = 725, zero = 707.5;
# 
EOData['ARI'] = (1 / EOData['B2']) - (1 / EOData['B4'])  # anthocyanin reflectance index, ARI1 = (1 / 550nm) - (1 / 700nm),
# #Non-Destructive Estimation of Anthocyanins and Chlorophylls in Anthocyanic Leaves (Gitelson, Chivkunova, Merzlyak);
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ari/

EOData['GNDVI'] = (EOData['B7'] - EOData['B2']) / (EOData['B7'] + EOData['B2']) #Green Normalized Difference Vegetation Index   (abbrv. GNDVI), General formula: (NIR - [540:570]) / (NIR + [540:570])
# 
EOData['MSI'] = EOData['B9'] / EOData['B7'] #Moisture Stress Index (abbrv. MSI), General formula: 1600nm / 820nm

# # ###############################################################################
EOData_med = EOData.groupby(['ID_1', 'Month'])['B1'].median().reset_index() #### any change here, should be done below
EOData_std = EOData.groupby(['ID_1', 'Month'])['B1'].std().reset_index() #### any change here, should be done below
# # Looping over bands to get median value for every plot (buffer) and every date
for i in range(1,len(band)):
    col_name = band[i]
    m = EOData.groupby(['ID_1', 'Month'])[col_name].median().reset_index()
    EOData_med = EOData_med.merge(m, on=['ID_1', 'Month'])
    
    m = EOData.groupby(['ID_1', 'Month'])[col_name].std().reset_index()
    EOData_std = EOData_std.merge(m, on=['ID_1', 'Month'])
    
EOData_med = pd.merge(EOData_med, DomSps, how='inner', on=['ID_1'])
EOData_std = pd.merge(EOData_std, DomSps, how='inner', on=['ID_1'])

EOData_med.sort_values(by=['Month'], inplace=True)
EOData_std.sort_values(by=['Month'], inplace=True)

EOData_med.to_csv(OutFile_med, sep=',', index=False)
EOData_std.to_csv(OutFile_std, sep=',', index=False)

#print('EOData_med')
#print(DomSps.head())
del EOData_med, EOData_std
# # # ##############################################################################

# # ##############################################################################
# # Looping over bands to get median value for every plot (buffer) and every date
EOData2019 = EOData[EOData['Year'] == 2019]
EOData_med = EOData2019.groupby(['ID_1', 'Month'])['B1'].median().reset_index() #### any change here, should be done below
EOData_std = EOData2019.groupby(['ID_1', 'Month'])['B1'].std().reset_index() #### any change here, should be done below
# # Looping over bands to get median value for every plot (buffer) and every date
for i in range(1,len(band)):
    col_name = band[i]
    m = EOData2019.groupby(['ID_1', 'Month'])[col_name].median().reset_index()
    EOData_med = EOData_med.merge(m, on=['ID_1', 'Month'])
    
    m = EOData2019.groupby(['ID_1', 'Month'])[col_name].std().reset_index()
    EOData_std = EOData_std.merge(m, on=['ID_1', 'Month'])
    
EOData_med = pd.merge(EOData_med, DomSps, how='inner', on=['ID_1'])
EOData_std = pd.merge(EOData_std, DomSps, how='inner', on=['ID_1'])

EOData_med.sort_values(by=['Month'], inplace=True)
EOData_std.sort_values(by=['Month'], inplace=True)

EOData_med.to_csv(OutFile2019_med, sep=',', index=False)
EOData_std.to_csv(OutFile2019_std, sep=',', index=False)

#print('EOData_med')
#print(DomSps.head())
del EOData_med, EOData_std
# # # ##############################################################################

# # ##############################################################################
# # Looping over bands to get median value for every plot (buffer) and every date
EOData2018 = EOData[EOData['Year'] == 2018]
EOData_med = EOData2018.groupby(['ID_1', 'Month'])['B1'].median().reset_index() #### any change here, should be done below
EOData_std = EOData2018.groupby(['ID_1', 'Month'])['B1'].std().reset_index() #### any change here, should be done below
# # Looping over bands to get median value for every plot (buffer) and every date
for i in range(1,len(band)):
    col_name = band[i]
    m = EOData2018.groupby(['ID_1', 'Month'])[col_name].median().reset_index()
    EOData_med = EOData_med.merge(m, on=['ID_1', 'Month'])
    
    m = EOData2018.groupby(['ID_1', 'Month'])[col_name].std().reset_index()
    EOData_std = EOData_std.merge(m, on=['ID_1', 'Month'])
    
EOData_med = pd.merge(EOData_med, DomSps, how='inner', on=['ID_1'])
EOData_std = pd.merge(EOData_std, DomSps, how='inner', on=['ID_1'])

EOData_med.sort_values(by=['Month'], inplace=True)
EOData_std.sort_values(by=['Month'], inplace=True)

EOData_med.to_csv(OutFile2018_med, sep=',', index=False)
EOData_std.to_csv(OutFile2018_std, sep=',', index=False)

#print('EOData_med')
#print(DomSps.head())
del EOData_med, EOData_std
# # # ##############################################################################

# # ##############################################################################
# # Looping over bands to get median value for every plot (buffer) and every date
EOData2020 = EOData[EOData['Year'] == 2020]
EOData_med = EOData2020.groupby(['ID_1', 'Month'])['B1'].median().reset_index() #### any change here, should be done below
EOData_std = EOData2020.groupby(['ID_1', 'Month'])['B1'].std().reset_index() #### any change here, should be done below
# # Looping over bands to get median value for every plot (buffer) and every date
for i in range(1,len(band)):
    col_name = band[i]
    m = EOData2020.groupby(['ID_1', 'Month'])[col_name].median().reset_index()
    EOData_med = EOData_med.merge(m, on=['ID_1', 'Month'])
    
    m = EOData2020.groupby(['ID_1', 'Month'])[col_name].std().reset_index()
    EOData_std = EOData_std.merge(m, on=['ID_1', 'Month'])
    
EOData_med = pd.merge(EOData_med, DomSps, how='inner', on=['ID_1'])
EOData_std = pd.merge(EOData_std, DomSps, how='inner', on=['ID_1'])

EOData_med.sort_values(by=['Month'], inplace=True)
EOData_std.sort_values(by=['Month'], inplace=True)

EOData_med.to_csv(OutFile2020_med, sep=',', index=False)
EOData_std.to_csv(OutFile2020_std, sep=',', index=False)

#print('EOData_med')
#print(DomSps.head())
del EOData_med, EOData_std
# # # ##############################################################################
