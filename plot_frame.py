import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from util import get_frame_at_timestamp, get_metadata
from geo_fun import depth_map_to_point_cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=' path to  the dataset to be loaded')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='name of sequence to be loaded')
    parser.add_argument('--desired_frame_time', type=float, required=True,
                        help='the in-sequence time of the frame to load [seconds]')
    args = parser.parse_args()

    metadata = get_metadata(args.dataset_path, args.seq_name)
    frame_rate = metadata['frame_rate']

    vid_file_name = args.seq_name + '_RGB'
    seq_out_path = os.path.join(args.dataset_path, args.seq_name)
    rgb_frame, _ = get_frame_at_timestamp(seq_out_path, args.desired_frame_time, vid_file_name, frame_rate,
                                          is_grayscale=True)
    plt.imshow(rgb_frame)
    plt.show()

    depth_vid_file_name = args.seq_name + '_Depth'
    depth_frame, _ = get_frame_at_timestamp(seq_out_path, args.desired_frame_time, depth_vid_file_name, frame_rate,
                                            is_grayscale=False)
    plt.imshow(depth_frame)
    plt.show()

    # The matrix of the camera intrinsics (camera_sys_hom_cord = cam_K_mat @ pixel_hom_cord)
    cam_K_mat = np.array([[metadata['fx'], 0, metadata['cx']],
                          [0, metadata['fy'], metadata['cy']],
                          [0, 0, 1]])

    surface_cord = depth_map_to_point_cloud(depth_frame, cam_K_mat)


if __name__ == '__main__':
    main()
