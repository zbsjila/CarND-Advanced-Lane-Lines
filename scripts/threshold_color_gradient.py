#!/usr/bin/env python
## import
import cv2
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import lanefindlib

## argin
if len(sys.argv) != 3:
    print("Sytnax error! Usage: ");
    print(sys.argv[0], "'../output_images/test*_undistort_warp.jpg' ../output_images/test_undistort_warp_pick_color.jpg");
    sys.exit(2);

img_fname_regex = sys.argv[1]; 
savefig_fname = sys.argv[2];

## imread
img_fname = sorted(glob.glob(img_fname_regex));
nimg = len(img_fname);

## loop
fig, ax = plt.subplots(8, nimg, sharex=True, sharey=True, figsize=(15, 10)); # 4/3 = 3/2.25; -> 2.5; 2.5*3
for ii in range(nimg):
    ### imread
    print("%d/%d, %s"%(ii, nimg, img_fname[ii]));

    ### choose color
    RGB = mpimg.imread(img_fname[ii]);

    ### call threshold_color_gradient
    SR_mask, S, S_mask, R, R_mask, Rsobelx, Rsobelx_mask = lanefindlib.threshold_color_gradient(RGB);
    
    ### imsave
    imsave_fname = img_fname[ii].replace('.jpg', '_mask.jpg'); 
    print("imsave", imsave_fname);
    mpimg.imsave(imsave_fname, SR_mask, cmap='gray');

    ### plot
    ax[0, ii].imshow(RGB);
    ax[0, ii].set_title(os.path.basename(img_fname[ii]));

    ax[1, ii].imshow(S);
    ax[1, ii].set_title('S');

    ax[2, ii].imshow(S_mask);
    ax[2, ii].set_title('S_mask');

    ax[3, ii].imshow(R);
    ax[3, ii].set_title('R');

    ax[4, ii].imshow(R_mask);
    ax[4, ii].set_title('R_mask');

    ax[5, ii].imshow(Rsobelx);
    ax[5, ii].set_title('Rsobelx');

    ax[6, ii].imshow(Rsobelx_mask);
    ax[6, ii].set_title('Rsobelx_mask');

    ax[7, ii].imshow(SR_mask);
    ax[7, ii].set_title('SR_mask');

## savefig
print("plt.savefig(%s)"%(savefig_fname));
plt.savefig(savefig_fname);

## show
plt.show()
