
This code loads the simulation output from the Simulator:  https://github.com/zsustc/colon_reconstruction_dataset 
that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
The simulator generates colonoscopic images with ground truth of camera poses.
The simulator has option for stereo cameras, but here we assume it was used with one camera.

The optical center of the first frame camera is used as the origin point of the global space.

!!!!!!!!
Camera calibration parameters: 
fx = 232.5044678; % unit in pixel fy = 232.5044678; cx = 240.0;
 cy = 320.0; baseline = 4.5; %unit in millimeter
!!!!!!!!!!!


## Pre-requisites
  * numpy 
  * matplotlib
  * h5py
  * [OpenCV](https://opencv.org/)

 
## How to run dataset generation
Run `generate_dataset.py` with the following required arguments:
* --sim_out_path: path to the simulator output (the folder containing the `Sequence_XXX` folders)
