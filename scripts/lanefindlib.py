#!/usr/bin/env python
"""
Include functions used in Project 2: Advanced Lane Finding
v 220122_143153: remove threshold_color threshold_gradient brightness_leveling
v 220122_213845: fill gap of dxR
v 220122_221757: fit: primary copied to secondary
v 220123_1418: primary: determiend by histogram peak counts instead of detected point counts (which is not reliable since there can be false detection)
220123_1502: fit left and right and detect Dx top/bottom differnece. If too big then use the dominate one; otherwise accept it is not parallel and accept each.
"""

## import
import numpy as np
import cv2
import matplotlib.pyplot as plt

## constants
### src_perspective dst_perspective: perspective transform
src_perspective = np.float32([
    [573, 466], # [599, 448], 
    [213, 720],	# 207 209
    [1105, 720], 
    [713, 466]]); # [682, 448] # 709 # 711
img_size = [1280, 720];
dst_perspective = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]]);

# the choice of dst_perspective determines the dimension of x/y pixel
dy_m = 30.0/img_size[1]; # 720 # meters per pixel in y dimension
dx_m = 3.7/(0.5*img_size[0]);# lane width: 3.7m. taking up half image width # 700 # meters per pixel in x dimension


## def calibrateCameraFromChessboardImages(images, nxCorners=8, nyCorners=6):
def calibrateCameraFromChessboardImages(images, nxCorners=8, nyCorners=6):
    ### objp
    # objp: object points in own coordinate of calibration chart like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0); can use physical units here
    objp = np.zeros((nyCorners*nxCorners,3), np.float32);
    objp[:,:2] = np.mgrid[0:nxCorners, 0:nyCorners].T.reshape(-1,2);

    ### findChessboardCorners -> objpoints imgpoints
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for idx, fname in enumerate(images): # [0:1]
        print("%d, %s findChessboardCorners/drawChessboardCorners"%(idx, fname));
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nxCorners,nyCorners), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nxCorners,nyCorners), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(1) # 0: wait forever # 500 wait too long; 10: very fast; no wait: no show
            #cv2.destroyAllWindows()
        else:
            print("FAILURE on this image!");

    ### calibrateCamera <- {objpoints imgpoints}
    img_size = (img.shape[1], img.shape[0])
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    """
    print("ret = ", ret);
    print("cameraMatrix = ", cameraMatrix);
    print("dist = ", dist);
    """
    return ret, cameraMatrix, dist;

## def perspectiveTransformMap(perspectiveTransform, xy_2col)
def perspectiveTransformMap(perspectiveTransform, xy_2col):
    """
    xy_2col -> xyMap_2col: coordinates of a point in the warped image
    Usage: dst_perspective_solved = lanefindlib.perspectiveTransformMap(perspectiveTransform, src);
    """
    ### xy_2col: pixel coordinates -- 2 col array [x y] or [[x1 y1] [x2 y2] ]
    if xy_2col.ndim == 1:
        xy_3col = np.append(xy_2col, 1.0);
    else:
        xy_3col = np.hstack((xy_2col, np.ones((xy_2col.shape[0], 1))));
    #xyMap_3col = (np.matmul(perspectiveTransform, xy_3col.T)).T;
    xyMap_3col = np.matmul(xy_3col, perspectiveTransform.T);
    if xy_2col.ndim == 1:
        xyMap_2col = np.hstack((xyMap_3col[0]/xyMap_3col[2], xyMap_3col[1]/xyMap_3col[2])); 
    else:
        xyMap_2col = (np.vstack((xyMap_3col[:, 0]/xyMap_3col[:, 2], xyMap_3col[:, 1]/xyMap_3col[:, 2]))).T; 
    return xyMap_2col;

## def undistort_warp
def undistort_warp(img, cameraMatrix, distCoeffs, perspectiveTransform):
    ### undistort
    undist = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix);
    
    ### warpPerspective
    undist_top_down = cv2.warpPerspective(undist, perspectiveTransform, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR);

    ### return
    return undist_top_down;

## def threshold_color_gradient(img):
def threshold_color_gradient(RGB, showFig=False):
    """ 
    RGB image -> combined_mask
    """

    ### const
    S_th = 177;
    R_th = 95; # 162;
    ksize = 9;
    Rsobelx_th = 12;

    ### color channel
    HLS = cv2.cvtColor(RGB, cv2.COLOR_RGB2HLS);
    S = HLS[:, :, 2];
    R = RGB[:, :, 0];
    Rsobelx = np.abs(cv2.Sobel(R, cv2.CV_64F, 1, 0, ksize=ksize));
    Rsobelx = np.uint8(255*Rsobelx/np.amax(Rsobelx));

    ### threshold
    S_mask = S >= S_th;
    R_mask = R >= R_th;
    Rsobelx_mask = Rsobelx > Rsobelx_th;
    Rsobelx_mask = cv2.erode(cv2.dilate((255*Rsobelx_mask).astype(np.uint8), np.ones((1, 15), np.uint8)), np.ones((1, 39), np.uint8)); # fill gap so it's not only edge but within as well

    ### combine
    #SR_mask = np.logical_and(np.logical_or(S_mask, R_mask), Rsobelx_mask);
    #SR_mask = np.logical_or(S_mask, Rsobelx_mask);
    SR_mask = np.logical_and(np.logical_or(S_mask, Rsobelx_mask), R_mask);

    ### clean up
    kernel_fill = np.ones((7, 3), np.uint8);  # (3, 7)
    SR_mask = cv2.dilate(cv2.erode((255*SR_mask).astype(np.uint8), kernel_fill), kernel_fill);

    ### return
    return SR_mask, S, S_mask, R, R_mask, Rsobelx, Rsobelx_mask; 

## def search_by_histogram_peak_sliding_window
def search_by_histogram_peak_sliding_window(binary_warped, nonzerox, nonzeroy):
    ### histogram peak
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.sum(binary_warped, axis=0)
    histogram = np.convolve(histogram, np.ones(151), mode="same");

    """
    fig, ax = plt.subplots();
    ax.plot(histogram);
    """

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    clearance = int(histogram.shape[0]//10) # search within [10% 90%]
    leftx_base = int(np.argmax(histogram[clearance:midpoint]) + clearance)
    rightx_base = int(np.argmax(histogram[midpoint:-clearance]) + midpoint)
    #print("histogram[leftx_base] = %d,  >= histogram[rightx_base] = %d"%(histogram[leftx_base], histogram[rightx_base]));

    """
    if histogram[leftx_base] >= histogram[rightx_base]:
        select_left = True;
    else:
        select_left = False;
    """

    ### hyperparameters of sliding window
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 75; # 50; # 100; # test5: reduce so the left lane won't be too much affected by false pixels
    # Set minimum number of pixels found to recenter window
    minpix = 250 # 50 # 

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)

    ### Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    ### Step through the windows one by one
    # img_pixels_sliding_window: draw sliding windows
    if binary_warped.ndim == 3 and binary_warped.shape[2] > 1:
        img_pixels_sliding_window = np.zeros_like(binary_warped);
    else:
        binary_warped = binary_warped.astype(np.uint8);
        binary_warped = np.uint8(binary_warped/np.amax(binary_warped)*255)
        img_pixels_sliding_window = np.dstack((binary_warped, binary_warped, binary_warped))

    for window in range(nwindows):
        #### Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)

        win_xleft_low = int(leftx_current - margin)  # Update this
        win_xleft_high = int(leftx_current + margin)  # Update this
        win_xright_low = int(rightx_current - margin)  # Update this
        win_xright_high = int(rightx_current + margin)  # Update this
        
        #### draw sliding windows on img_pixels_sliding_window
        #print("(win_xleft_low,win_y_low, win_xleft_high,win_y_high) = ", (win_xleft_low,win_y_low, win_xleft_high,win_y_high));
        #print("img_pixels_sliding_window.shape = ", img_pixels_sliding_window.shape);

        cv2.rectangle(img_pixels_sliding_window,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(255,255,255), 5) 
#        fig,ax = plt.subplots(1, 2, figsize=(8, 4)); 
#        ax[0].imshow(binary_warped);
#        ax[0].set_title("binary_warped");
#
#        ax[1].imshow(img_pixels_sliding_window);
#        ax[1].set_title("img_pixels_sliding_window");
#        plt.show();

        #cv2.rectangle(img_pixels_sliding_window, (230, 640), (430, 720), np.array([255,255,255], dtype=np.uint8), 5)
        #   rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img

        cv2.rectangle(img_pixels_sliding_window,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(255,255,255), 5) 
        
        #### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]; 
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]; 

        #print("%d/%d, [%d, %d], count = %d"%(window, nwindows, win_xright_low, win_xright_high, len(good_right_inds)), end='. ');
        
        #### Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #### recenter next window 
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]));
        if len(good_right_inds) > minpix:
            #print("nonzerox[good_right_inds] = ", nonzerox[good_right_inds]);
            rightx_current = int(np.mean(nonzerox[good_right_inds]));
            #print("rightx_current = ", rightx_current);
#        else:
#            print();

    ### Concatenate: list of lists -> array
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    ### return
    return left_lane_inds, right_lane_inds, img_pixels_sliding_window;

## def search_around_previous_fit
def search_around_previous_fit(binary_warped, left_fit, right_fit, nonzerox, nonzeroy):
    ### HYPERPARAMETER
    margin = 50; # 100

    ### search around fit lines for line pixels
    # Set the area of search based on activated x-values within the +/- margin of our polynomial function 
    nonzerox_left_fit = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2];
    left_lane_inds = ((nonzerox > nonzerox_left_fit - margin) & (nonzerox < nonzerox_left_fit + margin)).nonzero()[0]; # None

    nonzerox_right_fit = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2];
    right_lane_inds = ((nonzerox > nonzerox_right_fit - margin) & (nonzerox < nonzerox_right_fit + margin)).nonzero()[0]; # None

    return left_lane_inds, right_lane_inds;
    
## def xy_from_fit(height, left_fit, right_fit):
def xy_from_fit(height, left_fit, right_fit):
    ploty = np.linspace(0, height-1, height);
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty; 

## def find_lane_fit_refine: histogram peak + sliding window
def find_lane_fit_refine(binary_warped, left_fit_estimate=[], right_fit_estimate=[]):
    """
    left_fit_estimate, right_fit_estimate: initial guess, can be from last video frame.
    """
    ### nonzero x y positions 
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### search for lane pixels
    if len(left_fit_estimate) > 0 and len(right_fit_estimate) > 0:
        left_lane_inds, right_lane_inds = search_around_previous_fit(binary_warped, left_fit_estimate, right_fit_estimate, nonzerox, nonzeroy);
        img_pixels_sliding_window = np.dstack((binary_warped, binary_warped, binary_warped));
    else:
        left_lane_inds, right_lane_inds, img_pixels_sliding_window = search_by_histogram_peak_sliding_window(binary_warped, nonzerox, nonzeroy);

    ### lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ### fit update
    try:
        left_fit = np.polyfit(lefty, leftx, 2);
        right_fit = np.polyfit(righty, rightx, 2);

        left_fitx, right_fitx, ploty = xy_from_fit(binary_warped.shape[0], left_fit, right_fit);

        diffx_top_bottom = np.abs((right_fitx[-1] - left_fitx[-1]) - (right_fitx[0] - left_fitx[0]));  # difference in width between top and bottom
        """
        print("(left_fitx[0], right_fitx[0], left_fitx[-1], right_fitx[-1]) = ", (left_fitx[0], right_fitx[0], left_fitx[-1], right_fitx[-1]));
        print("diffx_top_bottom = ", diffx_top_bottom);
        print("0.1*binary_warped.shape[1] = ", 0.1*binary_warped.shape[1]);
        print("left_fit = ", left_fit);
        print("right_fit = ", right_fit);
        """

        if diffx_top_bottom > 0.1*binary_warped.shape[1]: # half width: normal width; 1/3 then too large differnce; cannot trust. use the dominate one 1/6 too large; needs to narrow down
            ### select left or right as primary to fit: ssd
            Dy_ssd_left = np.sqrt(np.sum((lefty - np.mean(lefty))**2));
            Dy_ssd_right = np.sqrt(np.sum((righty - np.mean(righty))**2));
            #print("Dy_ssd_left = ", Dy_ssd_left);
            #print("Dy_ssd_right = ", Dy_ssd_right);
            #if len(left_lane_inds) >= len(right_lane_inds):
            Dx0 = src_perspective[3, 0] - src_perspective[0, 0];
            Dx1 = src_perspective[2, 0] - src_perspective[1, 0];
            Dy = binary_warped.shape[0];

            if Dy_ssd_left > Dy_ssd_right:
                right_fit = left_fit.copy();
                #right_fit[2] = np.mean(rightx - (right_fit[0]*righty**2 + right_fit[1]*righty));
                weight_x = Dx0 + (Dx1 - Dx0)*np.float32(righty)/Dy;
                right_fit[2] = np.sum(weight_x*(rightx - (right_fit[0]*righty**2 + right_fit[1]*righty)))/np.sum(weight_x);
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            else:
                left_fit = right_fit.copy();
                #left_fit[2] = np.mean(leftx - (left_fit[0]*lefty**2 + left_fit[1]*lefty));
                weight_x = Dx0 + (Dx1 - Dx0)*np.float32(lefty)/Dy;
                left_fit[2] = np.sum(weight_x*(leftx - (left_fit[0]*lefty**2 + left_fit[1]*lefty)))/np.sum(weight_x);
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

            print("diffx_top_bottom > 0.1*binary_warped.shape[1], copy fit");
        """
        print("left_fit = ", left_fit);
        print("right_fit = ", right_fit);
        """

    except (ValueError, TypeError) as err:
        print("error = ", type(err));
        print("leftx = ", leftx);
        print("lefty = ", lefty);
        print("rightx = ", rightx);
        print("righty = ", righty);
        # Avoids an error if the above is not implemented fully
        #print("polyfit ValueError");
        left_fit = left_fit_estimate;
        right_fit = right_fit_estimate;
        pass

    ### return
    #fig, ax = plt.subplots(1, 1);
    #ax.imshow(img_pixels_sliding_window);
    #ax.set_title("img_pixels_sliding_window");

    #return leftx, lefty, rightx, righty, img_pixels_sliding_window
    return leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty, img_pixels_sliding_window;
    #return leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, left_fit, right_fit;
    #return left_fit, right_fit;

## def annotate_lane_pixels(mask_xy_lane_fit, leftx, lefty, rightx, righty):
def annotate_lane_pixels(mask_xy_lane_fit, leftx, lefty, rightx, righty):
    left_xy = np.vstack((leftx, lefty)).T;
    #left_xy_unwarp = np.int32(perspectiveTransformMap(perspectiveTransformInv, left_xy));
    right_xy = np.vstack((rightx, righty)).T;
    #right_xy_unwarp = np.int32(perspectiveTransformMap(perspectiveTransformInv, right_xy));

    mask_xy_lane_fit[left_xy[:, 1], left_xy[:, 0]] = [255, 0, 0];
    mask_xy_lane_fit[right_xy[:, 1], right_xy[:, 0]] = [234, 63, 247]; # [0, 0, 255]; # [0, 255, 0]; # [15, 60, 255]; # purple: not visible[106, 13, 173]; #[255, 0, 255]
    mask_xy_lane_fit = cv2.dilate(mask_xy_lane_fit, np.ones((1, 21), np.uint8));
    return mask_xy_lane_fit;

## def annotate_lane_fit(mask_xy_lane_fit, left_fit, right_fit):
def annotate_lane_fit(mask_xy_lane_fit, left_fitx, right_fitx, ploty): 

    ###
    height_warp = mask_xy_lane_fit.shape[0];

    xLaneCen = 0.5*(left_fitx[-1] + right_fitx[-1]);

    left_fit_xy = np.vstack((left_fitx, ploty)).T;
    right_fit_xy = np.vstack((right_fitx, ploty)).T;
    xy_fit = np.vstack((left_fit_xy, np.flipud(right_fit_xy)));

    ### xy_fit: fillPoly: fill before mark points so points are on top
    cv2.fillPoly(mask_xy_lane_fit, [np.int32(xy_fit)], (128, 128, 0)); # (255, 255, 0)); # 255 # (0, 128, 0) #

    return xLaneCen;

## def annotate_image: xy_lane; fit_coeffs -> img_annotated
def annotate_image(img, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty, perspectiveTransformInv, xCarWarped, width_warp, height_warp, extra_message='', verbose=True):
    ### fit_coeffs -> rcurv
    # _m: meters as units
    lefty_m = dy_m*lefty;
    leftx_m = dx_m*leftx;

    righty_m = dy_m*righty;
    rightx_m = dx_m*rightx;

    #ploty_m = dy_m*ploty;
    ym_eval = dy_m*(height_warp-1); # np.max(ploty_m);

    #print("left_fit = ", left_fit);
    Am_left = left_fit[0]/dy_m**2*dx_m;
    Bm_left = left_fit[1]/dy_m*dx_m;
    rcurv_left = (1.0 + (2.0*Am_left*ym_eval + Bm_left)**2)**1.5/2.0/np.abs(Am_left); 

    #print("right_fit = ", right_fit);
    Am_right = right_fit[0]/dy_m**2*dx_m;
    Bm_right = right_fit[1]/dy_m*dx_m;
    rcurv_right = (1.0 + (2.0*Am_right*ym_eval + Bm_right)**2)**1.5/2.0/np.abs(Am_right); 

    ### we trust the lane with more points
    if len(leftx) >= len(rightx):
        rcurv = rcurv_left;
    else:
        rcurv = rcurv_right; 
    #rcurv = 0.5*(rcurv_left + rcurv_right);

    ### fit_coeffs -> xy_fit 
    mask_xy_lane_fit = np.zeros((height_warp, width_warp, 3), dtype=np.uint8);
    xLaneCen = annotate_lane_fit(mask_xy_lane_fit, left_fitx, right_fitx, ploty);

    ### xy_fit -> xoffset
    xoffset = dx_m*(xCarWarped - xLaneCen);

    ### print
    if verbose:
        print("rcurv_left = %.0f m"%rcurv_left);
        print("rcurv_right = %.0f m"%rcurv_right);
        print("rcurv = %.0f m"%rcurv);
        print("xoffset = %05.2f m"%xoffset);

    ### {xy_lane xy_fit} -> xy_lane_fit_unwarp. mark then unwarp: avoid sparse points


    #left_fit_xy_unwarp = np.int32(perspectiveTransformMap(perspectiveTransformInv, left_fit_xy));
    #right_fit_xy_unwarp = np.int32(perspectiveTransformMap(perspectiveTransformInv, right_fit_xy));

    ### xy_lane_unwarp: mark with colors
    mask_xy_lane_fit = annotate_lane_pixels(mask_xy_lane_fit, leftx, lefty, rightx, righty);
    ### unwarp images (instead of points which cause sparcity)
    mask_xy_lane_fit_unwarped = cv2.warpPerspective(mask_xy_lane_fit, perspectiveTransformInv, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR);

    ### {img; xy_lane_fit_unwarp; rcurv; xoffset} -> img_annotated
    img_annotated = cv2.addWeighted(img, 1.0, mask_xy_lane_fit_unwarped, 0.5, 0); # [0.3 1.0]

    text1 = "Radius of Curvature = %5d m"%(rcurv); 
    if xoffset >= 0:
        left_or_right = "right";
    else:
        left_or_right = "left";
    text2 = "Vehicle is %.2f m %s of center"%(np.abs(xoffset), left_or_right);
    cv2.putText(img_annotated, extra_message, (20, 50), 0, 1.5, (255, 255, 255), thickness=2);
    cv2.putText(img_annotated, text1, (20, 100), 0, 1.5, (255, 255, 255), thickness=2);
    cv2.putText(img_annotated, text2, (20, 150), 0, 1.5, (255, 255, 255), thickness=2);

    ### return
    return img_annotated, mask_xy_lane_fit, mask_xy_lane_fit_unwarped;
    
## def fit_conditioning
def fit_conditioning(fit, fit_estimate): 
    if len(fit_estimate) == 0:
        fit_chosen = fit;
    else: 
        fit_deviation = np.abs(fit - fit_estimate);
        if fit_deviation[0] <= 0.0002 and fit_deviation[1] <= 0.2 and fit_deviation[2] <= 50: 
            fit_chosen = fit;
        else:
            print("anomaly: fit_deviation = ", fit_deviation);
            fit_chosen = fit_estimate;
    return fit_chosen;

## def update_latest(fit_chosen, fit_latest, counter):
def update_latest(fit_chosen, fit_latest, counter):
    k_smooth = 5; # window size of smoothing: average --> best fit for next frame
    if counter == 0:
        fit_latest = fit_chosen.reshape([1, 3]);
    elif counter > 0 and counter < k_smooth:
        fit_latest = np.concatenate((fit_latest, fit_chosen.reshape([1, 3])), axis=0);
    else: # >= k_smooth
        #print("counter%k_smooth, = ", counter%k_smooth);
        fit_latest[counter%k_smooth, :] = fit_chosen;
    fit_estimate = np.mean(fit_latest, axis=0); # median
    return fit_latest, fit_estimate; 


## def process_image_with_parameters
def process_image_with_parameters(img, cameraMatrix, distCoeffs, perspectiveTransform, perspectiveTransformInv, xCarWarped, left_fit_estimate=np.array([]), right_fit_estimate=np.array([]), left_fit_latest=np.array([[]]), right_fit_latest=np.array([[]]), counter=0, verbose=False, showSaveFig=True, output_prefix='', extra_message=''):

    ### go through the pipeline: undistort -> perspective -> threshold 
    undist_top_down = undistort_warp(img, cameraMatrix, distCoeffs, perspectiveTransform);

    # threshold
    mask_top_down = threshold_color_gradient(undist_top_down)[0];
    #mask_top_down = cv2.warpPerspective(combined_mask, perspectiveTransform, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR);
    
    ### histogram peak + sliding window  + poly fit
    leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty, img_pixels_sliding_window = find_lane_fit_refine(mask_top_down, left_fit_estimate=left_fit_estimate, right_fit_estimate=right_fit_estimate);

    ### fit_coeffs -> fit_coeffs_chosen -> {latest  -> estimate}
    left_fit_chosen = fit_conditioning(left_fit, left_fit_estimate);
    left_fit_latest, left_fit_estimate = update_latest(left_fit_chosen, left_fit_latest, counter);

    right_fit_chosen = fit_conditioning(right_fit, right_fit_estimate);
    right_fit_latest, right_fit_estimate = update_latest(right_fit_chosen, right_fit_latest, counter);

    print("counter = ", counter);
    """
    print("left_fit_latest = \n", left_fit_latest);
    print("right_fit_latest = \n", right_fit_latest);
    """
    print("left_fit = ", left_fit); 
    print("left_fit_estimate = ", left_fit_estimate);
    print("left_fit_chosen = ", left_fit_chosen); 
    print("right_fit = ", right_fit); 
    print("right_fit_estimate = ", right_fit_estimate);
    print("right_fit_chosen = ", right_fit_chosen); 

    ### left_fit_chosen -> left_fitx, right_fitx; here we don't update leftx... pixel detections but only the poly fit fill
    left_fitx, right_fitx, ploty = xy_from_fit(mask_top_down.shape[0], left_fit_chosen, right_fit_chosen);

    ### annotate_image;
    img_annotated, mask_xy_lane_fit, mask_xy_lane_fit_unwarped = annotate_image(img, leftx, lefty, rightx, righty, left_fit_chosen, right_fit_chosen, left_fitx, right_fitx, ploty, perspectiveTransformInv, xCarWarped, mask_top_down.shape[1], mask_top_down.shape[0], extra_message=extra_message, verbose=verbose);

    ### plot: undistort + threshold
    if verbose:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3), sharex=True, sharey=True);
        
        ax[0].imshow(img);
        ax[0].set_title('Input image');
        
        ax[1].imshow(undist_top_down);
        ax[1].set_title('undist_top_down');
        
        #ax[2].imshow(combined_mask);
        #ax[2].set_title('threshold');

    if showSaveFig:
        #savefig_fname = __file__.replace('.py', '_savefig_undistort_threshold.png');
        #savefig_fname = os.path.basename(__file__).replace('.py', '_savefig_undistort_threshold.png');
        savefig_fname = output_prefix + '_savefig_undistort_threshold.png';
        print("savefig", savefig_fname);
        plt.savefig(savefig_fname);
    
    ### plot: warped: mask_xy_lane_fit
    if verbose:
        fig_warped, ax_warped = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True);
        ax_warped[0, 0].imshow(undist_top_down);
        ax_warped[0, 0].set_title('undist_top_down');
        
        ax_warped[0, 1].imshow(mask_top_down);
        ax_warped[0, 1].set_title('mask_top_down');
        
        ax_warped[1, 0].imshow(img_pixels_sliding_window);
        ax_warped[1, 0].set_title('img_pixels_sliding_window');
        
        ax_warped[1, 1].imshow(mask_xy_lane_fit);
        ax_warped[1, 1].set_title('mask_xy_lane_fit');
        
    if showSaveFig:
        #savefig_fname = '../output_images/' + __file__.replace('.py', '_savefig_warped.png');
        savefig_fname = output_prefix + '_savefig_warped.png';
        print("savefig", savefig_fname);
        plt.savefig(savefig_fname);
    
    ### plot: warp back
    if verbose:
        fig_warpback, ax_warpback = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True);
        
        ax_warpback[0, 0].imshow(mask_xy_lane_fit);
        ax_warpback[0, 0].set_title("mask_xy_lane_fit");
        
        ax_warpback[0, 1].imshow(mask_xy_lane_fit_unwarped);
        ax_warpback[0, 1].set_title("mask_xy_lane_fit_unwarped");
        
        ax_warpback[1, 0].imshow(img);
        ax_warpback[1, 0].set_title("img");
        
        ax_warpback[1, 1].imshow(img_annotated);
        ax_warpback[1, 1].set_title("img_annotated");
        
    if showSaveFig:
        savefig_fname = output_prefix + '_savefig_warp_back.png'; # __file__.replace('.py', '_savefig_warp_back.png');
        print("savefig", savefig_fname);
        plt.savefig(savefig_fname);
    
    ### return
    return img_annotated, left_fit_latest, left_fit_estimate, right_fit_latest, right_fit_estimate;
