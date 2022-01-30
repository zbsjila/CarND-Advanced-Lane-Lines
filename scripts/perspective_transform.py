#!/usr/bin/env python
## import
import sys
import numpy as np
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import lanefindlib

## argin
if len(sys.argv) != 3:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "camera_data.p ../output_images/perspective_transform_savefig.png");
    sys.exit(1);
camera_data_pfile = sys.argv[1]; # "camera_calibration_output.p";
savefig_fname = sys.argv[2];

## imread
img1_fname = "../test_images/straight_lines1.jpg";
img1 = mpimg.imread(img1_fname); 
img2_fname = "../test_images/straight_lines2.jpg";
img2 = mpimg.imread(img2_fname); 

img_size = (img1.shape[1], img1.shape[0]);
print("img_size = ", img_size);

# undistort: lanefindlib.src_perspective points from original image so here we don't do undistort
print("load cameraMatrix distCoeffs from %s"%(camera_data_pfile));
camera_data_pickle = pickle.load(open(camera_data_pfile, "rb" ) ); 
cameraMatrix = camera_data_pickle["cameraMatrix"];
distCoeffs = camera_data_pickle["distCoeffs"];

print("undistort");
undist1 = cv2.undistort(img1, cameraMatrix, distCoeffs, None, cameraMatrix)
undist2 = cv2.undistort(img2, cameraMatrix, distCoeffs, None, cameraMatrix)

## plot: img vs undist1

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image')
#ax2.imshow(undist1)
#ax2.set_title('Undistorted and Warped Image')
##plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#undist_fname = img_fname.replace('.jpg', '_undist.jpg');
#print("imsave", undist_fname);
#mpimg.imsave(undist_fname, undist1);

## getPerspectiveTransform
perspectiveTransform = cv2.getPerspectiveTransform(lanefindlib.src_perspective, lanefindlib.dst_perspective);
print("perspectiveTransform = ", perspectiveTransform);

lanefindlib.dst_perspective_solved = lanefindlib.perspectiveTransformMap(perspectiveTransform, lanefindlib.src_perspective);
print("lanefindlib.dst_perspective= ", lanefindlib.dst_perspective);
print("lanefindlib.dst_perspective_solved = ", lanefindlib.dst_perspective_solved);

xyCar = np.float32((img_size[0]/2, img_size[1]));
xyCarWarped = lanefindlib.perspectiveTransformMap(perspectiveTransform, xyCar);
print("xyCar = ", xyCar);
print("xyCarWarped = ", xyCarWarped);
xCarWarped = xyCarWarped[0];
print("xCarWarped = ", xCarWarped);

## save in camera_data_pickle
print("dump perspectiveTransform in", camera_data_pfile);
#camera_data_pickle["dx_m"] = dx_m; 
#camera_data_pickle["dy_m"] = dy_m; 
camera_data_pickle["perspectiveTransform"] = perspectiveTransform; 
camera_data_pickle["xCarWarped"] = xCarWarped; 
pickle.dump(camera_data_pickle, open(camera_data_pfile, "wb"));

## warpPerspective -> top-down view
top_down1 = cv2.warpPerspective(undist1, perspectiveTransform, img_size, cv2.INTER_LINEAR);
top_down2 = cv2.warpPerspective(undist2, perspectiveTransform, img_size, cv2.INTER_LINEAR);
print("top_down1.shape = ", top_down1.shape);

## draw lanefindlib.src_perspective/lanefindlib.dst_perspective on undist1
cv2.polylines(undist1, [lanefindlib.src_perspective.astype(np.int32)], True, (255, 0, 0), thickness=5);
cv2.polylines(top_down1, [lanefindlib.dst_perspective.astype(np.int32)], True, (255, 0, 0), thickness=5);
cv2.polylines(undist2, [lanefindlib.src_perspective.astype(np.int32)], True, (255, 0, 0), thickness=5);
cv2.polylines(top_down2, [lanefindlib.dst_perspective.astype(np.int32)], True, (255, 0, 0), thickness=5);

## plot: img top_down1
#f, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(16, 10))
f, ax = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
f.tight_layout()
ax1, ax2, ax3, ax4 = ax.ravel();
ax1.imshow(undist1)
ax1.set_title('Undistorted image 1')
ax2.imshow(top_down1)
ax2.set_title('Undistorted and Warped Image 1')

ax3.imshow(undist2)
ax3.set_title('Undistorted image 2')
ax4.imshow(top_down2)
ax4.set_title('Undistorted and Warped Image 2')

## savefig
#savefig_fname = __file__.replace('.py', '_savefig.png');
print("savefig", savefig_fname);
plt.savefig(savefig_fname);
## end
plt.show()
