#!/usr/bin/env python
## import
import sys
import lanefindlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
import os

## argin
if len(sys.argv) != 4:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "../test_images/test1.jpg ../output_images/test1_undistort_warp.jpg camera_data.p");
    sys.exit(1);

img_fname = sys.argv[1]; 
imsave_fname = sys.argv[2]; 
camera_data_pfile = sys.argv[3]; # "camera_calibration_output.p";
if len(sys.argv) == 5:
    showSaveFig = int(sys.argv[4]);
else:
    showSaveFig = 1;

## imread
print("imread ", img_fname);
img = mpimg.imread(img_fname);

## load camera_data_pfile
camera_data_pickle = pickle.load(open(camera_data_pfile, "rb" ) ); 
cameraMatrix = camera_data_pickle["cameraMatrix"];
distCoeffs = camera_data_pickle["distCoeffs"];
perspectiveTransform = camera_data_pickle["perspectiveTransform"]; 

## process_image_with_parameters
print("undistort_warp");
undist_top_down = lanefindlib.undistort_warp(img, cameraMatrix, distCoeffs, perspectiveTransform);

## imsave
#img_annotated_fname = output_prefix + '_img_annotated.png'
print("imsave", imsave_fname);
mpimg.imsave(imsave_fname, undist_top_down);
