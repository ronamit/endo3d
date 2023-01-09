
This code loads the simulation output from the Simulator:  https://github.com/zsustc/colon_reconstruction_dataset 
that is described in "A Template-based 3D Reconstruction of Colon Structures and Textures from Stereo Colonoscopic Images", Zhang et al., IEEE Transactions on Medical Robotics  2022.
The simulator generates colonoscopic images with ground truth of camera poses.
The simulator has option for stereo cameras, but here we assume it was used with one camera.


[Example saved sequence](https://drive.google.com/drive/folders/1ADir7CwF9NTUVIH-1Og2BpBeAf10afYV?usp=sharing)

## Pre-requisites
  * numpy 
  * matplotlib
  * h5py
  * pandas
  * [OpenCV](https://opencv.org/)

 
## How to run dataset generation
Run `generate_dataset.py` with the following required arguments:
* --sim_out_path: path to the simulator output (the folder containing the `Sequence_XXX` folders)
* --frame_rate: frame retain Hz of the output videos, if 0 the frame rate will be extracted from the settings file

## Output description
* The camera_motion.csv file contains: pos_x, pos_y and pos_z coordinates of the camera center in the world space in millimeters,
and the [quaternion representations](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) of rotation from world space to the camera  space (quat_x, quat_y, quat_z, quat_w).
* the optical center of the first frame camera is used as the origin point of the world space.
* Note that Unity uses a left-handed coordinate system (see https://github.com/zsustc/colon_reconstruction_dataset)