 Simulator code:  https://github.com/zsustc/colon_reconstruction_dataset

This dataset contains 15 cases of simulated stereo colonoscopic images with ground truth of camera poses
 (left camera poses, right camera poses and rotations).
In each folder named Case#, its subfolder "left" contains the left camera images and "right" contains
the right camera images.
And three "txt" for mat files are left camera poses, right camera poses and rotations, the optical center
of the first frame camera is used as the origin point of the global space.

Camera calibration parameters: fx = 232.5044678; % unit in pixel fy = 232.5044678; cx = 240.0;
 cy = 320.0; baseline = 4.5; %unit in millimeter


## Pre-requisites
* Local [ffmpeg executable](https://ffmpeg.org/download.html)
     * For Mac with ARM architecture:
      **[download ffmpeg](https://www.osxexperts.net/) and put the ffmpeg file a some local path. 
     * Set it as executable with $ chmod 755 ffmpeg
     * For Windows:
      [download ffmpeg](https://ffmpeg.org/download.html)
      and put in a some local location


## How to run dataset generation
Run `generate_dataset.py` with the following required arguments:
* --ffmpeg_path: path to ffmpeg executable
* --sim_out_path: path to the simulator output (the folder containing the `Sequence_XXX` folders)
* --output_path: path to output folder