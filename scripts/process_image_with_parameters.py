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
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "../test_images/test1.jpg ../output_images/test1 camera_data.p [showSaveFig=1]");
    sys.exit(1);
img_fname = sys.argv[1]; 
output_prefix = sys.argv[2]; 
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
perspectiveTransformInv = np.linalg.inv(perspectiveTransform);
#dx_m = camera_data_pickle["dx_m"]; 
#dy_m = camera_data_pickle["dy_m"]; 
xCarWarped = camera_data_pickle["xCarWarped"]; 

## process_image_with_parameters
print("process_image_with_parameters");
img_annotated = lanefindlib.process_image_with_parameters(img, cameraMatrix, distCoeffs, perspectiveTransform, perspectiveTransformInv, xCarWarped, verbose=True, showSaveFig=True, output_prefix=output_prefix)[0];

## imsave
img_annotated_fname = output_prefix + '_img_annotated.png'
print("imsave", img_annotated_fname);
mpimg.imsave(img_annotated_fname, img_annotated);

## end
if showSaveFig:
    plt.show();
