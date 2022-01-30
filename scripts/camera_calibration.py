#!/usr/bin/env python
## import
import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)

"""
from os.path import dirname, abspath
lanefindlib_dir = dirname(dirname(dirname(abspath(__file__)))) + '/scripts';
lanefindlib_dir = "../../scripts" #print("lanefindlib_dir = ", lanefindlib_dir);
sys.path.insert(1, lanefindlib_dir);
"""
import lanefindlib 

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

## argin
if len(sys.argv) != 5:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "'../camera_cal/calibration*.jpg' camera_data.p ../camera_cal/calibration5.jpg ../output_images/calibration5_undistort_savefig.png");
    print("Notice that single quote for argv[1] is important!");
    sys.exit(1);
images_regex = sys.argv[1];
camera_data_pfile = sys.argv[2]; # "camera_calibration_output.p";
img_test_fname = sys.argv[3];
img_test_savefig_fname = sys.argv[4];

## run calibrateCameraFromChessboardImages
images = glob.glob(images_regex) # list of strings # '../camera_cal/calibration*.jpg'
# 9*5, 9*6, 9*6
ret, cameraMatrix, distCoeffs =  lanefindlib.calibrateCameraFromChessboardImages(images, 9, 6); # tested: need to be exact; otherwise will fail
print("cameraMatrix = ", cameraMatrix);
print("distCoeffs = ", distCoeffs);

## save cameraMatrix distCoeffs for future undistort
camera_data_pickle = {};
camera_data_pickle["cameraMatrix"] = cameraMatrix; 
camera_data_pickle["distCoeffs"] = distCoeffs; 
print("dump", camera_data_pfile);
pickle.dump(camera_data_pickle, open(camera_data_pfile, "wb"));

## undistort 
img_test = cv2.imread(img_test_fname);
img_test_undistort = cv2.undistort(img_test, cameraMatrix, distCoeffs, None, cameraMatrix)

## show
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
ax1.imshow(img_test)
ax1.set_title('Original Image')
ax2.imshow(img_test_undistort)
ax2.set_title('Undistorted Image')

#img_test_savefig_fname = '../output_images/calibration5_undistort_savefig.png'; # __file__.replace('.py', '_savefig.png');
print("savefig", img_test_savefig_fname);
plt.savefig(img_test_savefig_fname);

## end
plt.show();
