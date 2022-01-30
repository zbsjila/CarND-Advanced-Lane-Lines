#!/usr/bin/env python
## import
import numpy as np
import matplotlib.pyplot as plt
import sys

## argin
if len(sys.argv) != 6:
    print("Sytnax error! Usage: ");
    print(sys.argv[0], "../log_video_output/left_fit.txt ../log_video_output/left_fit_estimate.txt ../log_video_output/right_fit.txt ../log_video_output/right_fit_estimate.txt ../log_video_output/plot_fit_and_estimate.pdf"); 
    sys.exit(2);
left_fit_txt = sys.argv[1];
left_fit_estimate_txt = sys.argv[2];
right_fit_txt = sys.argv[3];
right_fit_estimate_txt = sys.argv[4];
savefig_fname = sys.argv[5];

## loadtxt
left_fit = np.loadtxt(left_fit_txt);
left_fit_estimate = np.loadtxt(left_fit_estimate_txt);
right_fit = np.loadtxt(right_fit_txt);
right_fit_estimate = np.loadtxt(right_fit_estimate_txt);

## plot
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(15, 9));

# coeff 0
ax[0, 0].plot(left_fit[:, 0], 'r.', label='left_fit[0]')
ax[0, 0].plot(left_fit_estimate[:, 0], '--r', label='left_fit_estimate[0]')
ax[0, 0].plot(right_fit[:, 0], 'b.', label='right_fit')
ax[0, 0].plot(right_fit_estimate[:, 0], '--b', label='right_fit_estimate[0]')
ax[0, 1].legend();

ax[0, 1].plot(left_fit[:, 0] - left_fit_estimate[:, 0], '-r.', label='Diff left_fit[:, 0]');
ax[0, 1].plot(right_fit[:, 0] - right_fit_estimate[:, 0], '-b.', label='Diff right_fit[:, 0]');
ax[0, 1].legend();

# coeff 1
ax[1, 0].plot(left_fit[:, 1], '-r.', label='left_fit[1]')
ax[1, 0].plot(left_fit_estimate[:, 1], '-r', label='left_fit_estimate[1]')
ax[1, 0].plot(right_fit[:, 1], '-b.', label='right_fit')
ax[1, 0].plot(right_fit_estimate[:, 1], '-b', label='right_fit_estimate[1]')
ax[1, 0].legend();

ax[1, 1].plot(left_fit[:, 1] - left_fit_estimate[:, 1], '-r.', label='Diff left_fit[:, 1]');
ax[1, 1].plot(right_fit[:, 1] - right_fit_estimate[:, 1], '-b.', label='Diff right_fit[:, 1]');
ax[1, 1].legend();

# coeff 2
ax[2, 0].plot(left_fit[:, 2], '-r.', label='left_fit[2]')
ax[2, 0].plot(left_fit_estimate[:, 2], '-r', label='left_fit_estimate[2]')
ax[2, 0].plot(right_fit[:, 2], '-b.', label='right_fit')
ax[2, 0].plot(right_fit_estimate[:, 2], '-b', label='right_fit_estimate[2]')
ax[2, 0].legend();

ax[2, 1].plot(left_fit[:, 2] - left_fit_estimate[:, 2], '-r.', label='Diff left_fit[:, 2]');
ax[2, 1].plot(right_fit[:, 2] - right_fit_estimate[:, 2], '-b.', label='Diff right_fit[:, 2]');
ax[2, 1].legend();

#plt.legend();

## savefig
print("savefig", savefig_fname);
plt.savefig(savefig_fname);

## show
plt.show()
