'''
Simulator code:  https://github.com/zsustc/colon_reconstruction_dataset

# This dataset contains 15 cases of simulated stereo colonoscopic images with ground truth of camera poses
# (left camera poses, right camera poses and rotations).
In each folder named Case#, its subfolder "left" contains the left camera images and "right" contains
the right camera images.
And three "txt" for mat files are left camera poses, right camera poses and rotations, the optical center
of the first frame camera is used as the origin point of the global space.

Camera calibration parameters: fx = 232.5044678; % unit in pixel fy = 232.5044678; cx = 240.0;
 cy = 320.0; baseline = 4.5; %unit in millimeter
'''

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir = r'/Users/ronamit/PublicDatasets/ColonSim/generated_cases'
cases_paths = glob.glob(data_dir + '/*/')
print(cases_paths)

# camera intrinsic parameters
focal_length = 4.969783  # [millimeter]
sensor_width = 10.26  # [millimeter]
sensor_height = 7.695  # [millimeter]
cols = 320  # [pixels] image width
rows = 240  # [pixels]  image height
fx = focal_length * cols / sensor_width  # focal length in x-axis [pixels]
fy = focal_length * rows / sensor_height  # focal length in y-axis [pixels]
cx = cols / 2.0  # middle of the image in x-axis [pixels]
cy = rows / 2.0  # middle of the image in y-axis [pixels]


def load_case(case_path):
    rgb_img_paths = glob.glob(case_path + '/*.png')
    for img_path in rgb_img_paths:
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.show()
        pass





    # depth_exr{i} = exrread(['path\SUK_L_depth',num2str(i,'%05d'),'.exr']); end
    #
    # scan_gt = cell(1,num);
    #
    # depth_scale = 5.0;
    #
    # for i = 1:num
    #
    # xyzPoints = zeros(rows, cols, 3);
    #
    # for v = 1:rows % rows



for case_path in cases_paths:
    load_case(case_path)
