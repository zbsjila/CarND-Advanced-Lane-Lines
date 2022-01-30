#!/usr/bin/env python
## import
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
import sys
import os
import glob
import lanefindlib

## argin
if len(sys.argv) != 3:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "../output_images/test*_undistort_warp_mask.jpg ../output_images/test_undistort_warp_mask_search.jpg");
    sys.exit(2);

img_fname_regex = sys.argv[1];
savefig_fname = sys.argv[2];

## imread
img_fname = sorted(glob.glob(img_fname_regex));
nimg = len(img_fname);

## call
fig, ax = plt.subplots(3, nimg, sharex=True, sharey=True, figsize=(15, 7.5));
for ii in range(nimg): # 1): # [3]: # 
    binary_warped = mpimg.imread(img_fname[ii]);
    if binary_warped.ndim > 2 and binary_warped.shape[2] > 1:
        binary_warped = binary_warped[:, :, 0];

    leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty, img_pixels_sliding_window = lanefindlib.find_lane_fit_refine(binary_warped);

    img_lane_pixels = np.dstack((binary_warped, binary_warped, binary_warped)); # np.zeros_like(binary_warped);
    lanefindlib.annotate_lane_pixels(img_lane_pixels, leftx, lefty, rightx, righty);
    lanefindlib.annotate_lane_fit(img_lane_pixels, left_fitx, right_fitx, ploty);

    ### plot
    ax[0, ii].imshow(binary_warped);
    ax[0, ii].set_title(os.path.basename(img_fname[ii]).split('_')[0]);

    ax[1, ii].imshow(img_pixels_sliding_window);
    ax[1, ii].set_title("img_pixels_sliding_window");

    ax[2, ii].imshow(img_lane_pixels);
    ax[2, ii].set_title("img_lane_pixels");

## savefig
print("plt.savefig", savefig_fname);
plt.savefig(savefig_fname);

## show
plt.show()
