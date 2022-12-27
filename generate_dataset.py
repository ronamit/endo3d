'''

'''

import os
import glob
import argparse
from pathlib import Path
import shutil
from util import frames2video
parser = argparse.ArgumentParser()
parser.add_argument('--ffmpeg_path', type=str, required=True, help='path to ffmpeg executable')
parser.add_argument('--sim_out_path', type=str, required=True,
                    help=' path to the simulator output (the folder containing the `Sequence_XXX` folders)')
parser.add_argument('--dataset_out_path', type=str, default='./dataset_out', help='path to the output dataset')
args = parser.parse_args()

sim_out_path = Path(args.sim_out_path)
seq_paths = glob.glob(args.sim_out_path / '*/')

print(seq_paths)

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

dataset_out_path = Path(args.dataset_out_patj).resolve()
if not os.path.isdir(dataset_out_path):
    os.makedirs(dataset_out_path)
shutil.copy2(sim_out_path / 'MySettings.set', dataset_out_path / 'MySettings.set')

for seq_path in seq_paths:
    seq_name = os.path.split(seq_path)[-2]
    frame_rate = 30
    frames2video(images_path=seq_path,
                 ffmpeg_path=args.ffmpeg_path,
                 dataset_out_path=dataset_out_path,
                 vid_file_name=seq_name,
                 frame_rate=frame_rate)


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


