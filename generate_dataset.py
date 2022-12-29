'''

'''

import argparse
import glob
import os
import shutil

from util import create_rgb_video, save_depth_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg_path', type=str, required=True, help='path to ffmpeg executable')
    parser.add_argument('--sim_out_path', type=str, required=True,
                        help=' path to the simulator output (the folder containing the `Sequence_XXX` folders)')
    args = parser.parse_args()

    sim_out_path = os.path.abspath(args.sim_out_path)
    dataset_out_path = sim_out_path + '_datasets'

    if not os.path.isdir(dataset_out_path):
        os.makedirs(dataset_out_path)

    seq_paths = glob.glob(sim_out_path + '/*/')
    print(seq_paths)

    for seq_in_path in seq_paths:
        seq_name = os.path.basename(os.path.normpath(seq_in_path))
        seq_out_path = os.path.join(dataset_out_path, seq_name)
        if not os.path.isdir(seq_out_path):
            os.makedirs(seq_out_path)
        shutil.copy2(os.path.join(seq_in_path, 'MySettings.set'), os.path.join(seq_out_path, 'MySettings.set'))

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

        frame_rate = 20  # shotPerSec":"float(20)

        save_depth_frames(seq_in_path=seq_in_path,
                          seq_out_path=seq_out_path,
                          vid_file_name=seq_name + '_depth',
                          frame_rate=frame_rate)

        create_rgb_video(seq_in_path=seq_in_path,
                         seq_out_path=seq_out_path,
                         vid_file_name=seq_name + '_rgb',
                         frame_rate=frame_rate,
                         ffmpeg_path=args.ffmpeg_path)

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
