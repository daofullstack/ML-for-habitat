##############################################################################
'''
Script to convert H5 files into one CSV file.
H5 files are produced after running 01_ExtractBandValues.py
Author : Suvarna M. Punalekar
Date : 6 Sept 2021
'''
##############################################################################
import glob, os, sys
import glob, os, sys
from os import path
import numpy as np
import h5py
import pandas as pd
import datetime

root = 'data/LW_Analysis/SeminaturalMapping_ver3/SemiNatural_BandExtract/'
HDFList = sorted(glob.glob(root+'*.h5'))
#HDFList = HDFList[:1]

df = pd.DataFrame(columns=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'ID_1', 'Date'])

for Input in HDFList:
    try:
        hdf = h5py.File(Input, 'r') # 'w' for write, 'a' for append/edit
        array = hdf.get('/DATA/DATA')[()]
        hdf.close()
        print(array.shape)
        
    except Exception:
        raise SystemExit('Error: unable to read input file.')
    
    array = array[array[:,0]!= 0, :]
    
    df_array = pd.DataFrame(array, columns=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'ID_1'])
    
    date_str = Input.split('/')[-1]
    df_array["Date"] = date_str.split('_')[1]
    
    df = df.append(df_array, ignore_index = True)
    
    del df_array

print(df)
    
#df.to_csv(root + 'GLAMA_buffer_S2bands2019.csv', sep='\t')
    
df.to_csv(root + 'SemiNatSps_BandExtract_8Tiles_2015to2020.csv', sep=',')
