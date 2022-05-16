#!/usr/bin/env/python3
'''
Description: A script for applying the developed model (.sav) on the input images to create maps
Scrpt for SemiNat and Cultivated map
Authors: Pete Bunting and Suvarna Punalekar (Aberystwyth University).
Date : 26 Oct 2021
'''

import numpy

from osgeo import gdal

import rsgislib.imageutils
import rsgislib.imagecalc
import rsgislib.rastergis
import rsgislib
from rsgislib.classification import ClassSimpleInfoObj
import pandas as pd
import glob, os, sys
import os.path
import sys
import shutil
import random

import h5py

from rios import applier
from rios import cuiprogress
from rios import rat


def apply_sklearn_classifer(classTrainInfo, skClassifier, imgMask, imgMaskVal, imgFileInfo, outputImg, gdalformat,
                            classClrNames=True, outScoreImg=None):
    """
This function uses a trained classifier and applies it to the provided input image.

:param classTrainInfo: dict (where the key is the class name) of rsgislib.classification.ClassSimpleInfoObj
                       objects which will be used to train the classifier (i.e., train_sklearn_classifier()),
                       provide pixel value id and RGB class values.
:param skClassifier: a trained instance of a scikit-learn classifier
                     (e.g., use train_sklearn_classifier or train_sklearn_classifer_gridsearch)
:param imgMask: is an image file providing a mask to specify where should be classified. Simplest mask is all
                the valid data regions (rsgislib.imageutils.genValidMask)
:param imgMaskVal: the pixel value within the imgMask to limit the region to which the classification is applied.
                   Can be used to create a heirachical classification.
:param imgFileInfo: a list of rsgislib.imageutils.ImageBandInfo objects (also used within
                    rsgislib.imageutils.extractZoneImageBandValues2HDF) to identify which images and bands are to
                    be used for the classification so it adheres to the training data.
:param outputImg: output image file with the classification. Note. by default a colour table and class names column
                  is added to the image. If an error is produced use HFA or KEA formats.
:param gdalformat: is the output image format - all GDAL supported formats are supported.
:param classClrNames: default is True and therefore a colour table will the colours specified in classTrainInfo
                      and a ClassName column (from imgFileInfo) will be added to the output file.
:param outScoreImg: A file path for a score image. If None then not outputted.

    """
    out_score_img = False
    if outScoreImg is not None:
        out_score_img = True
    
    infiles = applier.FilenameAssociations()
    infiles.imageMask = imgMask
    numClassVars = 0
    for imgFile in imgFileInfo:
        infiles.__dict__[imgFile.name] = imgFile.fileName
        numClassVars = numClassVars + len(imgFile.bands)

    outfiles = applier.FilenameAssociations()
    outfiles.outimage = outputImg
    if out_score_img:
        outfiles.out_score_img = outScoreImg
    otherargs = applier.OtherInputs()
    otherargs.classifier = skClassifier
    otherargs.mskVal = imgMaskVal
    otherargs.numClassVars = numClassVars
    otherargs.n_classes = len(classTrainInfo)
    otherargs.imgFileInfo = imgFileInfo
    otherargs.out_score_img = out_score_img

    try:
        import tqdm
        progress_bar = rsgislib.TQDMProgressBar()
    except:
        progress_bar = cuiprogress.GDALProgressBar()

    aControls = applier.ApplierControls()
    aControls.progress = progress_bar
    aControls.drivername = gdalformat
    aControls.omitPyramids = True
    aControls.calcStats = False

    # RIOS function to apply classifer
    def _applySKClassifier(info, inputs, outputs, otherargs):
        """
        Internal function for rios applier. Used within applyClassifer.
        """
        outClassVals = numpy.zeros_like(inputs.imageMask, dtype=numpy.uint32)
        #print(outClassVals.shape)
        
        if otherargs.out_score_img:
            outScoreVals = numpy.zeros((otherargs.n_classes, inputs.imageMask.shape[1], inputs.imageMask.shape[2]), dtype=numpy.float32)
            #print(outScoreVals.shape)


        if numpy.any(inputs.imageMask == otherargs.mskVal):
            outClassVals = outClassVals.flatten()
            if otherargs.out_score_img:
                outScoreVals = outScoreVals.reshape(outClassVals.shape[0], otherargs.n_classes)
                #print(outScoreVals.shape)
            imgMaskVals = inputs.imageMask.flatten()
            classVars = numpy.zeros((outClassVals.shape[0], otherargs.numClassVars), dtype=numpy.float)
            # Array index which can be used to populate the output array following masking etc.
            ID = numpy.arange(imgMaskVals.shape[0])
            classVarsIdx = 0
            for imgFile in otherargs.imgFileInfo:
                imgArr = inputs.__dict__[imgFile.name]
                for band in imgFile.bands:
                    classVars[..., classVarsIdx] = imgArr[(band - 1)].flatten()
                    classVarsIdx = classVarsIdx + 1
            classVars = classVars[imgMaskVals == otherargs.mskVal]
            ID = ID[imgMaskVals == otherargs.mskVal]
            predClass = otherargs.classifier.predict(classVars)
            outClassVals[ID] = predClass
            outClassVals = numpy.expand_dims(outClassVals.reshape((inputs.imageMask.shape[1], inputs.imageMask.shape[2])), axis=0)
            if otherargs.out_score_img:
                predClassScore = otherargs.classifier.predict_proba(classVars)
                #print(predClassScore.shape)
                outScoreVals[ID] = predClassScore
                outScoreVals = outScoreVals.T
                outScoreVals = outScoreVals.reshape((otherargs.n_classes, inputs.imageMask.shape[1], inputs.imageMask.shape[2]))
        outputs.outimage = outClassVals
        if otherargs.out_score_img:
            outputs.out_score_img = outScoreVals

    print("Applying the Classifier")
    applier.apply(_applySKClassifier, infiles, outfiles, otherargs, controls=aControls)
    print("Completed")
    rsgislib.rastergis.populateStats(clumps=outputImg, addclrtab=True, calcpyramids=True, ignorezero=True)
    if out_score_img:
        rsgislib.imageutils.popImageStats(outScoreImg, usenodataval=True, nodataval=0, calcpyramids=True)
    
    if classClrNames:
        ratDataset = gdal.Open(outputImg, gdal.GA_Update)
        red = rat.readColumn(ratDataset, 'Red')
        green = rat.readColumn(ratDataset, 'Green')
        blue = rat.readColumn(ratDataset, 'Blue')
        ClassName = numpy.empty_like(red, dtype=numpy.dtype('a255'))
        ClassName[...] = ''

        for classKey in classTrainInfo:
            print("Apply Colour to class \'" + classKey + "\'")
            red[classTrainInfo[classKey].id] = classTrainInfo[classKey].red
            green[classTrainInfo[classKey].id] = classTrainInfo[classKey].green
            blue[classTrainInfo[classKey].id] = classTrainInfo[classKey].blue
            ClassName[classTrainInfo[classKey].id] = classKey

        rat.writeColumn(ratDataset, "Red", red)
        rat.writeColumn(ratDataset, "Green", green)
        rat.writeColumn(ratDataset, "Blue", blue)
        rat.writeColumn(ratDataset, "ClassName", ClassName)
        ratDataset = None

###################################################################################################
import rsgislib.classification
import rsgislib.imageutils
import joblib
#import pickle

InDir = 'data/Sentinel2/LW_Analysis/SpectralIndices/Composites/WalesComposites/'
InList = sorted(glob.glob(InDir + 'Wales*adj.kea'))

InList.append('data/DEM/Wales_LIDAR_Derived_DTM_10m.kea')
InList.append('data/DEM/Wales_LIDAR_Derived_Slope_10m.kea')

print(len(InList))

ModelDir = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver3/Cultivated_SemiNat_CompositeExtract_Plot600/Model_S2medTerrain_feat15/'
model = ModelDir + 'featurenum_15_model.sav'
skclf = joblib.load(open(model, 'rb'))

featnum = 'featnum_15'
# Output classification image
out_cls_img = ModelDir + featnum + '_CultSemiNat_class_S2medTer_2018to2019_Wales.kea'
# Output probability image
out_proba_img = ModelDir +  featnum + '_CultSemiNat_proba_S2medTer_2018to2019_Wales.kea'
# Input mask image
img_msk = 'data/Sentinel2/LW_Analysis/SeminaturalMapping_ver3/Wales_ExtentMask_ignorenan.kea'

imgs_info = []

# Selected features

Features = pd.read_csv(ModelDir + 'Selected_features.csv', sep=',', index_col=False)
Features = Features[featnum].tolist()

fileList = []
for Feature in Features:

    for file in InList:
        filen = file.split('/')[-1]
        tag1 = filen.split('_')[1]
        tag2 = filen.split('_')[2]
        tag3 = filen.split('_')[3]
        tag4 = filen.split('_')[4]
        tag = tag1 + tag2 + tag3 + tag4
        #print(tag, Feature)
        if '.' in tag:
            tag = tag.split('.')[0]
        if tag == Feature:
            print(tag, Feature)
            imgs_info.append(rsgislib.imageutils.ImageBandInfo(fileName=file, name=Feature, bands=[1]))
            fileList.append(file)
print(fileList)
# 
print(len(fileList))
# #print(imgs_info)
# # 
#Define final colours rather than the randomly allocated from get_class_training_data.
cls_train_info = dict()

cls_train_info['1'] = rsgislib.classification.ClassSimpleInfoObj(id=1, fileH5=None, red=50, green=130, blue=240)
cls_train_info['2'] = rsgislib.classification.ClassSimpleInfoObj(id=2, fileH5=None, red=70, green=150, blue=220)


#apply_sklearn_classifer(cls_train_info, rf_mdl, 'mosaic_example_vmsk_sub.kea', 1, imgs_info, './cls_img_sub.kea', 'KEA', classClrNames=True, outScoreImg=None)#outScoreImg='cls_score_img_sub.kea')

#Apply classification
apply_sklearn_classifer(cls_train_info, skclf, img_msk, 1, imgs_info, out_cls_img, 'KEA', classClrNames=True, outScoreImg=out_proba_img)
