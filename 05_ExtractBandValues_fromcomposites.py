##############################################################################
'''
Script to extract band values from composites of S2 images, corresponding to the sampling polygons
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

#root = 'data/LW_S2ARD/*/'
root = 'data/Sentinel2/LW_Analysis/SpectralIndices/Composites/WalesComposites/'

outpath1 = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver3/tmp/'
outpath2 = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver3/SemiNatural_CompositeExtract_ver3Plots/'
##############################################################################

DTM = 'data/DEM/Wales_LIDAR_Derived_DTM_10m.tif'
Slope = 'data/DEM/Wales_LIDAR_Derived_Slope_10m.tif'
##############################################################################
ImageList = []

print(root)
ImageList = sorted(glob.glob(root + 'Wales_*_2016to2019_*_adj.kea'))
#ImageList = sorted(glob.glob(root + 'Wales_FebtoApr_*_2016to2019_*_adj.kea'))
print(ImageList[:5])
print('length of imagelist after S2 files')
print(len(ImageList))
##############################################################################
ImageListVH = sorted(glob.glob(root + 'Wales_*_VH_2019_*_adj.kea'))
ImageListVV = sorted(glob.glob(root + 'Wales_*_VV_2019_*_adj.kea'))
ImageListVHVV = sorted(glob.glob(root + 'Wales_*_VHVV_2019_*_adj.kea'))
##############################################################################

ImageList = ImageList + ImageListVH + ImageListVV  + ImageListVHVV
print('length of imagelist after S1 files')
print(len(ImageList))


ImageList.append(DTM)
ImageList.append(Slope)
print('length of imagelist after S2 + S1 + dem + slope')
print(len(ImageList))
##############################################################################
VectorFile = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver3/TrainingData_ver3Plots_id.shp'
#VectorFile = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver2/Species_training/Species_buff30m_id.shp'
##############################################################################
# 
# #Rasterize the buffer shapefile 
# # Open Shapefile
Shapefile = ogr.Open(VectorFile)
Shapefile_layer = Shapefile.GetLayer()
# Get drivers for the raster file
#for image in ImageList:
numb = 0
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
    #OutHDF = outpath2 + OutHDF[:34]
    OutHDF = outpath2 + OutHDF[:-4]
    OutHDF = OutHDF+ '.h5'
    print(OutHDF)
    
    fileInfo = []
    fileInfo.append(rsgislib.imageutils.ImageBandInfo(image, 'Image1', [1]))
    fileInfo.append(rsgislib.imageutils.ImageBandInfo(OutImage2, 'Image2', [1]))
    rsgislib.imageutils.extractZoneImageBandValues2HDF(fileInfo, OutImage1, OutHDF, 1.0)
    
    os.remove(OutImage1)
    os.remove(OutImage2)
    numb = numb + 1

print("Done!!")


