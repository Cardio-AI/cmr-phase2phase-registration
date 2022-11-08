# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:18:45 2017

@author: Sandy Engelhardt

Computation of the 17 Segments AHA Model according to 
Manuel D. Cerqueira et al., "Standardized Myocardial Segmentation and Nomenclature
for Tomographic Imaging of the Heart", Circulation 2002
"""

import SimpleITK as sitk
import numpy as np


#labels as defined in the challenge data
lRV = 1
lMyo = 2
lLV = 3

def cart2pol(x, y,newCenterX, newCenterY):
    x = x - newCenterX
    y = y - newCenterY
    
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    
    if(theta < 0):
        theta += 2 * np.pi
    
    return theta, r

def pol2cart(theta, r,newCenterX, newCenterY):   
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
     # old origin 
    x = x + newCenterX
    y = y + newCenterY
    
    x = np.floor(x)
    y = np.floor(y)
    
    return x, y

       
def convertCartCOS(y, npimage):
    y = np.abs(y - npimage.shape[1])
    return y
    

def getCenterMyo(sitkimage):
    
    #compute centroid of myocard on this slice
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitkimage)
    centerMyo = stats.GetCentroid(lMyo)  
    return centerMyo
    

def isRVdefinedOnThisSlice(npimage,z):
    return np.any(npimage[z,:,:] == lRV)

def computeRVMyoIntersection(npimage):
    
    sitkimage = sitk.GetImageFromArray(npimage)                            
    centerMyo = getCenterMyo(sitkimage)
    yNew = convertCartCOS(centerMyo[1],npimage)
    
    #---------- detect the intersection between RV and Myo----------------

    # reset image region of LV to label 2 instead of 3
    # such that left heart becomes one region
    sitkimage -= (sitkimage>2)*1
    # add higher value to this region, to distinguish it on the Edge map
    sitkimage += (sitkimage>1)*10
                   
    #cast is required by filter
    sitkimage = sitk.Cast( sitkimage, sitk.sitkFloat32 )
    #filter highlights the edges outer edges
    EdgefilteredImage = sitk.SobelEdgeDetection(sitkimage)
    #sitk.WriteImage(EdgefilteredImage, "sitk_output_Edge.nrrd")

    # so we have only hightlighted the contour where rv and background meets
    binaryimageRV = sitk.BinaryThreshold(EdgefilteredImage,1,5)

    polarThetaMax = -1
    polarThetaMin = 361
    for y in range(binaryimageRV.GetSize()[1]):
        for x in range(binaryimageRV.GetSize()[0]):
            pixelval = binaryimageRV.GetPixel(x,y)
            if (pixelval == 1):
                y1 = convertCartCOS(y,npimage)
                polar= cart2pol(x,y1,centerMyo[0],yNew)
                if(polar[0] < polarThetaMin):
                    polarThetaMin = polar[0]
                if(polar[0] > polarThetaMax):
                    polarThetaMax = polar[0]
            
    #print(np.degrees(polarThetaMin))
    #print(np.degrees(polarThetaMax))
    
    return (polarThetaMin, polarThetaMax)
    
     
    
def compute6Segments(npimage, SliceNumber, intersections):
     
    polarThetaMax = intersections[1]
    polarThetaMin = intersections[0]
    
    sitkimageIn = sitk.GetImageFromArray(npimage)
    sitkimageOut = sitk.GetImageFromArray(npimage)

    midangle = (polarThetaMax - polarThetaMin)/2.0

    # -------insert in angle array in sorted order (segment separations in polar coordinates (just theta))
    # -------sorting is done counterclockwise according to the labeling in the AHA-Paper 
    #--------this is just for basal and mid-cavity slices

    angles = np.zeros(6) #in radiant
    
    angles[0] = polarThetaMax - np.pi 
    angles[1] = polarThetaMin      
    angles[2] = polarThetaMax - midangle      
    angles[3] = polarThetaMax
    angles[4] = polarThetaMin - np.pi 
    angles[5] = (polarThetaMax - midangle) - np.pi
         

    # otherwise we get negative angles (just add +2pi)
    angles += (angles<0)*2*np.pi
              
    angles.sort()
   
    itemindex = np.where(angles==polarThetaMin)
    start = itemindex[0]-1 # index where to start labeling
    saveStartValue = angles[start]
                
    angles -= angles[start] # rotate polar coordinate system!
    angles += (angles<0)*2*np.pi
    
    angles.sort()
          
    centerMyo = getCenterMyo(sitkimageIn)  
    yNew = convertCartCOS(centerMyo[1],npimage)    
    #print(np.degrees(angles))

    for y in range(sitkimageIn.GetSize()[1]):
        for x in range(sitkimageIn.GetSize()[0]):
            pixelval = sitkimageIn.GetPixel(x,y)
            if (pixelval == lMyo):
                y1 = convertCartCOS(y,npimage)
                polar = cart2pol(x, y1, centerMyo[0],yNew)
                Theta1 = polar[0]- saveStartValue
                if(Theta1 < 0):
                    Theta1+=2*np.pi
                index = 0
                i = 0
                while i < len(angles):
                    if(Theta1 > angles[i]):
                        index = i 
                    i+=1  

                Region = int((SliceNumber-1)*6)
                sitkimageOut.SetPixel(x,y,lMyo*100 + (index+1)+Region)
                      
    npimageOut = sitk.GetArrayFromImage(sitkimageOut)
    type(npimageOut)
    
    return npimageOut


def compute4Segments(npimage, SliceNumber, intersections):
     
    # should be almost the same here
    polarThetaMax = intersections[1]
    polarThetaMin = intersections[0]
    
    sitkimageIn = sitk.GetImageFromArray(npimage)
    sitkimageOut = sitk.GetImageFromArray(npimage)

    midOfIntersection = ((polarThetaMax - polarThetaMin)/2.0) + polarThetaMin

    # -------insert in angle array in sorted order (segment separations in polar coordinates (just theta))
    # -------sorting is done counterclockwise according to the labeling in the AHA-Paper 
    #--------this is just for basal and mid-cavity slices

    angles = np.zeros(4) #in radiant
    
    angles[0] = midOfIntersection - (np.pi/2.0)     
    angles[1] = midOfIntersection      
    angles[2] = midOfIntersection + (np.pi/2.0)
    angles[3] = midOfIntersection - np.pi 

    # otherwise we get negative angles (just add +2pi)
    angles += (angles<0)*2*np.pi
             
    start = 3 # index where to start labeling
    saveStartValue = angles[start]
                     
    angles -= angles[start] # rotate polar coordinate system!
    angles += (angles<0)*2*np.pi
    
    angles.sort()
          
    centerMyo = getCenterMyo(sitkimageIn)
    yNew = convertCartCOS(centerMyo[1],npimage)     
    
   # print(np.degrees(angles))

    for y in range(sitkimageIn.GetSize()[1]):
        for x in range(sitkimageIn.GetSize()[0]):
            pixelval = sitkimageIn.GetPixel(x,y)
            if (pixelval == lMyo):
                y1 = convertCartCOS(y,npimage)
                polar = cart2pol(x, y1, centerMyo[0],yNew)
                Theta1 = polar[0]- saveStartValue
                if(Theta1 < 0):
                    Theta1+=2*np.pi
                index = 0
                i = 0
                while i < len(angles):
                    if(Theta1 > angles[i]):
                        index = i 
                    i+=1  

                Region = int((SliceNumber-1)*6)
                sitkimageOut.SetPixel(x,y,lMyo*100 + (index+1)+Region)
                   
    npimageOut = sitk.GetArrayFromImage(sitkimageOut)
    type(npimageOut)
    
    return npimageOut     
      

# input = 3D volume as npimage
def assignSliceLabel(npimage):
    
    lv_mask = (npimage==lRV)
    myo_mask = (npimage==lRV)
    most_apical_lv = np.max(np.where(lv_mask)[0])
    most_apical_myo = np.max(np.where(myo_mask)[0])

    if(most_apical_lv < most_apical_myo):
        print("apex slice with myocard is available!!! (uncommon)")
        apexSliceAvailable = True
    else:
        #should be more common in ACDC collection
         apexSliceAvailable = False

    inputimage = sitk.GetImageFromArray(npimage)
    #%% compute Bounding Box
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(inputimage)

    #compute bounding box of myocard
    bb=stats.GetBoundingBox(lMyo) 
    
    if(apexSliceAvailable):
        averageSizeInZ = (bb[5]-1) / 3
    else:
         averageSizeInZ = bb[5]/3
    
    sizeZ_apicalRegion = np.ceil(averageSizeInZ)
    sizeZ_midCavityRegion = np.rint(averageSizeInZ)
    sizeZ_BasalRegion = np.floor(averageSizeInZ)
  
    #assign each slice (in the bounding box) to a label according to its region

    sliceLabel = np.zeros(npimage.shape[0])
    
    i = IndexMostBasalSlice = bb[2]
    while(i < (IndexMostBasalSlice+sizeZ_BasalRegion)):
       sliceLabel[i] = 1 #basal 
       i+=1
       
    while(i < (IndexMostBasalSlice+sizeZ_BasalRegion+sizeZ_midCavityRegion)):
       sliceLabel[i] = 2 #midcav 
       i+=1
       
    while(i < (IndexMostBasalSlice+sizeZ_BasalRegion+sizeZ_midCavityRegion+sizeZ_apicalRegion)):
      sliceLabel[i] = 3 #apical 
      i+=1
      
    if(apexSliceAvailable):
      sliceLabel[i] = 4 #apex
    
    #print(sliceLabel)
    return sliceLabel     



def computeSegmentsOfMyo(npimage):
    
    #---------- specific for these datasets are the following points:
    # heart is captured in a short axes slice stack
    # most basal slice is always at the bottom
    # most apical slice is always at the top
    # the apex of the myocard is often not visible and won't be analysed 
    
    rv_mask = (npimage==lRV)
    most_apical_rv = np.max(np.where(rv_mask)[0])
    min_basal_rv = np.min(np.where(rv_mask)[0])
        
    slicesLabel = assignSliceLabel(npimage)
    
    #copy it to output volume
    npimageVolSeg=np.copy(npimage)
    
    # if RV is not defined on a slice, propagate intersections from these slices
    npimage1 = npimage[most_apical_rv,:,:]
    intersections_most_apical_rv = computeRVMyoIntersection(npimage1)
    
    npimage1 = npimage[min_basal_rv,:,:]
    intersections_min_basal_rv = computeRVMyoIntersection(npimage1)
    
    for z in range(npimage.shape[0]):
       if(slicesLabel[z] > 0): # 0 is background slice
          #print("current slice= " + str(z) + " "+ str(isRVdefinedOnThisSlice(npimage,z)))
          # ExtractSlices
          npimage1 = npimage[z,:,:]
          if(isRVdefinedOnThisSlice(npimage,z)):
              intersections = computeRVMyoIntersection(npimage1)
              if(slicesLabel[z] in range(1,3)):
                  npimageSeg = compute6Segments(npimage1,slicesLabel[z],intersections)
              elif(slicesLabel[z] == 3):
                  npimageSeg = compute4Segments(npimage1,slicesLabel[z],intersections)  
          else:
              if(slicesLabel[z] in range(1,3)):
                  npimageSeg = compute6Segments(npimage1,slicesLabel[z],intersections_min_basal_rv)
              elif(slicesLabel[z] == 3):
                  npimageSeg = compute4Segments(npimage1,slicesLabel[z],intersections_most_apical_rv)  
          #write slice back to volume
          npimageVolSeg[z,:,:] = npimageSeg
                       
    # label 4 == apex is not evaluated at the moment!
    return npimageVolSeg
    

# path="C:/ACDC_Challenge/training/"
# end = "_gt.nii.gz"
#
# fn=[
# "patient001/patient001_frame01"
# ]
#
# endOut = "_seg.nrrd"
#
# for name in fn:
#     filename = path + name + end
#     sitkinputimage = sitk.ReadImage(filename)
#     npimage = sitk.GetArrayFromImage(sitkinputimage)
#     type(npimage)
#
#     npimageVolSeg = computeSegmentsOfMyo(npimage)
#     sitkimageVolSeg = sitk.GetImageFromArray(npimageVolSeg)
#     sitkimageVolSeg.CopyInformation(sitkinputimage)
#     newname =  name
#     newname = newname.replace("/","-")
#     filenameOut = newname + endOut
#     sitk.WriteImage(sitkimageVolSeg, filenameOut)

path = '/mnt/ssd/julian/data/raw/flowfields/v10_nn011_nopostprocess/hh_20190621/'
name =
filename = path + name
sitkinputimage = sitk.ReadImage(filename)



