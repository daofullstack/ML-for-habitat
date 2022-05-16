##############################################################################
'''
Script to convert H5 files into one CSV file. It reads all composite indices pixel values and copies in the CSV file.
H5 files are produced after running 05_ExtractBandValues_fromcomposites
Author : Suvarna M. Punalekar
Date : 6 Spet 2021
'''
##############################################################################

import glob, os, sys
import glob, os, sys
from os import path
import numpy as np
import h5py
import pandas as pd
import datetime

root = 'data/LW_Analysis/SeminaturalMapping_ver3/SemiNatural_CompositeExtract_ver3Plots/'
HDFList = sorted(glob.glob(root+'*.h5'))

   
df = pd.DataFrame()

numb = 0
for Input in HDFList:
    try:
        hdf = h5py.File(Input, 'r') # 'w' for write, 'a' for append/edit
        array = hdf.get('/DATA/DATA')[()]
        hdf.close()
        print(array.shape)
        
    except Exception:
        raise SystemExit('Error: unable to read input file.')
    
    #array = array[array[:,0]!= 0, :]
    
    
    Input = Input.split('/')[-1]
    tag1 = Input.split('_')[1]
    tag2 = Input.split('_')[2]
    tag3 = Input.split('_')[3]
    tag4 = Input.split('_')[4]
    
    tag4 = tag4.split('.')[0]
    
    tag = tag1 + tag2 + tag3 + tag4
    
    tagid = 'ID_1'
    
    print(tag)
    df_array = pd.DataFrame(array, columns=[tag, tagid])
    #print(df_array.head())
    
    if numb == 0:
        df = df.append(df_array)
    else:
        df_array.index = df.index
        df[[tag]] = df_array[[tag]]
    
    numb = numb + 1
        
    del df_array
    del tag, tag1, tag2, tag3, tag4, tagid

print(df.tail())
df = df.loc[:,~df.columns.duplicated()]
df['ID_1'] = df['ID_1'].astype(int)
print(df.tail())

#df.to_csv(root + 'GLAMA_buffer_S2bands2019.csv', sep='\t')
    
df.to_csv(root + 'SemiNatural_S1S2Terrain_ver3Plots.csv', sep=',')
#     
    

