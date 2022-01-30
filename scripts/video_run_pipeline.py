#!/usr/bin/env python
## import
from moviepy.editor import VideoFileClip
import sys
import lanefindlib
import pickle
import numpy as np

## constants

## argin
if len(sys.argv) != 4:
    print("Syntax error! Usage: ");
    print(sys.argv[0], "../project_video.mp4 ../project_video_output.mp4 camera_data.p");
    sys.exit(1)

video_input_fname = sys.argv[1]; # "test_videos/solidWhiteRight.mp4";
video_output_fname = sys.argv[2];
camera_data_pfile = sys.argv[3];

## video input 
clip1 = VideoFileClip(video_input_fname);

## load parameters
print("load", camera_data_pfile);
camera_data_pickle = pickle.load(open(camera_data_pfile, "rb" ) ); 
cameraMatrix = camera_data_pickle["cameraMatrix"];
distCoeffs = camera_data_pickle["distCoeffs"];
perspectiveTransform = camera_data_pickle["perspectiveTransform"]; 
perspectiveTransformInv = np.linalg.inv(perspectiveTransform);
#xm_per_pix = camera_data_pickle["xm_per_pix"]; 
#ym_per_pix = camera_data_pickle["ym_per_pix"]; 
xCarWarped = camera_data_pickle["xCarWarped"]; 

## init

## def process_image
def process_image(img):

    ### run process_image_with_parameters
    extra_message = "frame # %5d"%(process_image.counter - 1);  # somehow I need to subtract 1 to be right
    img_annotated, process_image.left_fit_latest, process_image.left_fit_estimate, process_image.right_fit_latest, process_image.right_fit_estimate = lanefindlib.process_image_with_parameters(img, cameraMatrix, distCoeffs, perspectiveTransform, perspectiveTransformInv, xCarWarped, left_fit_estimate=process_image.left_fit_estimate, right_fit_estimate=process_image.right_fit_estimate, left_fit_latest=process_image.left_fit_latest, right_fit_latest=process_image.right_fit_latest, counter=process_image.counter, verbose=False, showSaveFig=False, extra_message=extra_message);

    ### next
    process_image.counter += 1;

    ### return
    return img_annotated;

## init
process_image.counter = 0; # -1; # this way frame [0, N-1] matched with ffmpeg
process_image.left_fit_estimate = np.array([]);
process_image.right_fit_estimate = np.array([]);
process_image.left_fit_latest = np.array([[]]);
process_image.right_fit_latest = np.array([[]]);

## fl_image
print("fl_image");
white_clip = clip1.fl_image(process_image); #NOTE: this function expects color images!!

## write_videofile
print("write_videofile %s"%video_output_fname);
white_clip.write_videofile(video_output_fname, audio=False)
