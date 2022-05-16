##############################################################################
'''
Script to extract band values from pixels of S2 images, corresponding to the sampling polygons
Author : Suvarna M. Punalekar
Date : 6 Sept 2021
'''
##############################################################################
import glob, os, sys
from os import path
import rasterio
import numpy as np
from osgeo import gdal, gdal_array, ogr
import rsgislib
import rsgislib.imagecalibration
import rsgislib.imageutils

root = 'data/LW_S2ARD/'
outpath1 = 'data/LW_Analysis/SeminaturalMapping_ver3/tmp/'
outpath2 = 'data/LW_Analysis/SeminaturalMapping_ver3/SemiNatural_BandExtract/'

#ImageList = []

f = open(root + 'S2_UUC_UVC_UVD_UUE_UVE_FileList_2015to2019_filtered_Jul2020.txt')
print(root + 'S2_UUC_UVC_UVD_UUE_UVE_FileList_2015to2019_filtered_Jul2020.txt')
ImageList1 = f.read().splitlines()
print(len(ImageList1))
del f

f = open(root + 'S2_FileList_2020_filtered_Jul2021.txt')
print(root + 'S2_FileList_2020_filtered_Jul2021.txt')
ImageList2 = f.read().splitlines()
print(len(ImageList2))
del f

ImageList = ImageList1 + ImageList2
print(len(ImageList))

ImageList = ImageList[240:]

VectorFile = 'data/LW_Analysis/SeminaturalMapping_ver3/TrainingData_890plots_id.shp'
# 
# #Rasterize the buffer shapefile 
# # Open Shapefile
Shapefile = ogr.Open(VectorFile)
Shapefile_layer = Shapefile.GetLayer()
# Get drivers for the raster file
#for image in ImageList:
numb = 240
for image in ImageList:
    print(numb)
    image_ds = gdal.Open(image)
    
    OutImage = image.split('/')[-1]
    OutImage = outpath1 + OutImage
    OutImage1 = OutImage[:-4] + '_buffer1.kea'
    #print(OutImage1)
    Output = gdal.GetDriverByName('KEA').Create(OutImage1, image_ds.RasterXSize, image_ds.RasterYSize, 1, 1)
    Output.SetProjection(image_ds.GetProjectionRef())
    Output.SetGeoTransform(image_ds.GetGeoTransform())
    
    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    #gdal.RasterizeLayer(Output, [1], Shapefile_layer, options = ["ATTRIBUTE=Grasscod_1"])
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[1])
    del Output

    OutImage2 = OutImage[:-4] + '_buffer2.kea'
    #print(OutImage2)
    Output2 = gdal.GetDriverByName('KEA').Create(OutImage2, image_ds.RasterXSize, image_ds.RasterYSize, 1, 2)
    Output2.SetProjection(image_ds.GetProjectionRef())
    Output2.SetGeoTransform(image_ds.GetGeoTransform())
    # Write data to band 1
    Band = Output2.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(Output2, [1], Shapefile_layer, options = ["ATTRIBUTE=ID_1"])
    del Output2
    
    OutHDF = image.split('/')[-1]
    OutHDF = outpath2 + OutHDF[:34]
    OutHDF = OutHDF+ '.h5'
    print(OutHDF)
    
    fileInfo = []
    fileInfo.append(rsgislib.imageutils.ImageBandInfo(image, 'Image1', [1,2,3,4,5,6,7,8,9,10]))
    fileInfo.append(rsgislib.imageutils.ImageBandInfo(OutImage2, 'Image2', [1]))
    rsgislib.imageutils.extractZoneImageBandValues2HDF(fileInfo, OutImage1, OutHDF, 1.0)
    
    os.remove(OutImage1)
    os.remove(OutImage2)
    numb = numb + 1

print("Done!!")


