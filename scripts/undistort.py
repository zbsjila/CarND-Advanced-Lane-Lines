#!/usr/bin/env python
## import
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import cv2

## argin
if len(sys.argv) != 4:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "camera_data.p ../test_images/test1.jpg ../output_images/test1_savefig_undistort.png");
    sys.exit(1);

calibration_pickle_fname = sys.argv[1];
img_fname = sys.argv[2];
savefig_fname = sys.argv[3];

## load cameraMatrix distCoeffs 
print("load cameraMatrix distCoeffs from", calibration_pickle_fname);
calibration_pickle = pickle.load(open(calibration_pickle_fname, "rb"));
cameraMatrix = calibration_pickle["cameraMatrix"]; 
distCoeffs = calibration_pickle["distCoeffs"];

## undistort 
print("imread", img_fname);
img = mpimg.imread(img_fname); # cv2
img_undistort = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix);

## show
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
ax1.imshow(img)
ax1.set_title('Original Image')
ax2.imshow(img_undistort)
ax2.set_title('Undistorted Image')

## savefig
figname = savefig_fname; # __file__.replace('.py', '_savefig.png');
print("savefig", figname);
plt.savefig(figname);

## end
plt.show();
