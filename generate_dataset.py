'''

'''

import argparse
import glob
import json
import os
import shutil

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt

from util import get_seq_id, find_between_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_out_path', type=str, required=True,
                        help=' path to the Unity simulator output')
    args = parser.parse_args()

    sim_out_path = os.path.abspath(args.sim_out_path)
    dataset_path = sim_out_path + '_Dataset'
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    print(f'Datasets will be saved to {dataset_path}')
    seq_paths = glob.glob(sim_out_path + '/*/')
    print('Simulation sequences to be loaded: ', seq_paths)

    for seq_in_path in seq_paths:
        print('Loading: ', seq_in_path)
        seq_name = os.path.basename(os.path.normpath(seq_in_path))
        seq_out_path = os.path.join(dataset_path, seq_name)
        if not os.path.isdir(seq_out_path):
            os.makedirs(seq_out_path)

        metadata = save_metadata(seq_in_path, seq_out_path, seq_name)
        frame_rate = metadata['frame_rate']

        save_depth_frames(seq_in_path=seq_in_path,
                          seq_out_path=seq_out_path,
                          vid_file_name=seq_name + '_Depth',
                          frame_rate=frame_rate,
                          depth_vid_scale=metadata['depth_vid_scale'])

        create_rgb_video(seq_in_path=seq_in_path,
                         seq_out_path=seq_out_path,
                         vid_file_name=seq_name + '_RGB',
                         frame_rate=frame_rate)



def save_metadata(seq_in_path, seq_out_path, seq_name):
    sim_settings_path = os.path.join(seq_in_path, 'MySettings.set')
    shutil.copy2(sim_settings_path, os.path.join(seq_out_path, 'Sim_GUI_Settings.set'))  # copy the settings file
    # Extract the settings from the settings file:
    camFOV_deg = find_between_str(sim_settings_path, r'"camFOV":"float\(', r'\)"')
    camFOV_deg = float(camFOV_deg)  # camera FOV [deg]
    camFOV_rad = np.deg2rad(camFOV_deg)  # camera FOV [rad]
    frame_width = find_between_str(sim_settings_path, r'"shotResX":"float\(', r'\)"')
    frame_width = int(frame_width)  # [pixels]
    frame_height = find_between_str(sim_settings_path, r'"shotResY":"float\(', r'\)"')
    frame_height = int(frame_height)  # [pixels]
    frame_rate = find_between_str(sim_settings_path, r'"shotPerSec":"float\(', r'\)"')
    frame_rate = float(frame_rate)  # [Hz]

    # Manually set parameters:
    image_plane_width = 10.26  # [millimeter]  # according to https://github.com/zsustc/colon_reconstruction_dataset
    image_plane_height = 7.695  # [millimeter] # according to https://github.com/zsustc/colon_reconstruction_dataset
    depth_vid_scale = 10.0  # [millimeter]  # the depth video values should be multiplied by depth_vid_scale to get
    # the depth in millimeter (the value was chosen to spread the depth values in the range of 0-255)

    sensor_radius = 0.5 * np.sqrt(image_plane_width ** 2 + image_plane_height ** 2)  # [millimeter]
    focal_length = sensor_radius / np.tan(camFOV_rad / 2.0)  # [millimeter]
    sx = image_plane_width / frame_width  # the with of each pixel's sensor [millimeter/pixel]
    sy = image_plane_height / frame_height  # the height of each pixel's sensor [millimeter/pixel]
    fx = focal_length / sx  # [pixels]
    fy = focal_length / sy  # [pixels]
    cx = frame_width / 2.0  # middle of the image in x-axis [pixels]
    cy = frame_height / 2.0  # middle of the image in y-axis [pixels]

    metadata = {'focal_length': focal_length,
                'image_plane_width': image_plane_width,
                'image_plane_height': image_plane_height,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'sx': sx,
                'sy': sy,
                'frame_rate': frame_rate,
                'depth_vid_scale': depth_vid_scale}
    metadata_path = os.path.join(seq_out_path, seq_name + '_metadata.json')
    with open(metadata_path, 'w', ) as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)
    return metadata


def create_rgb_video(seq_in_path, seq_out_path, vid_file_name, frame_rate):
    """
    Convert a sequence of images to a video using ffmpeg
    """

    output_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'
    seq_id = get_seq_id(seq_in_path)
    frames_paths = glob.glob(os.path.join(seq_in_path, f'{seq_id}_*.png'))
    frames_paths.sort()
    print(f'Number of RGB frames: {len(frames_paths)}')
    frame_size = cv2.imread(frames_paths[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out_video = cv2.VideoWriter(filename=output_path,
                                fourcc=fourcc,
                                fps=frame_rate,
                                frameSize=frame_size,
                                isColor=True)
    for frame_path in frames_paths:
        im = cv2.imread(frame_path)
        out_video.write(im)
    out_video.release()
    print(f'Video saved to: {output_path}')


def save_depth_frames(seq_in_path, seq_out_path, vid_file_name, frame_rate, depth_vid_scale, limit_frame_num=0,
                      save_h5_file=False):
    """
    Load a sequence of depth images from a folder
    """

    seq_id = get_seq_id(seq_in_path)
    depth_files_paths = glob.glob(os.path.join(seq_in_path, f'{seq_id}_depth*.exr'))
    depth_files_paths.sort()
    if limit_frame_num:
        # for debugging
        depth_files_paths = depth_files_paths[:limit_frame_num]
    n_frames = len(depth_files_paths)
    print(f'Number of depth frames: {n_frames}')

    # Open the output file for writing
    output_path = os.path.join(seq_out_path, vid_file_name)

    # Compute the size
    depth_img = cv2.imread(depth_files_paths[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    frame_size = depth_img.shape[:2]

    depth_frames = np.zeros((n_frames, frame_size[0], frame_size[1]), dtype=np.float32)

    # Iterate over the EXR files and add them to the video
    for i_frame, exr_path in enumerate(depth_files_paths):
        # All 3 channels are the same (depth), so we only need to read one
        depth_img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        depth_frames[i_frame] = depth_img

    min_depth = np.min(depth_frames.flatten())
    max_depth = np.max(depth_frames.flatten())
    print(f'Min-depth: {min_depth}, Max-depth: {max_depth}')
    assert min_depth > 0, 'Min depth should be positive'
    assert max_depth < 255 / depth_vid_scale, 'Max depth should be smaller than 255 / depth_vid_scale'
    plt.hist(depth_frames.flatten(), bins='auto')
    plt.savefig(output_path + '_histogram.png')
    plt.close()

    # Save as matrix
    if save_h5_file:
        with h5py.File(output_path + '.h5', 'w') as hf:
            hf.create_dataset(vid_file_name, data=depth_frames, compression='gzip')
        print(f'Depth frames saved to: {output_path}.h5')

    save_depth_video(depth_frames, output_path, frame_rate, depth_vid_scale)


def save_depth_video(depth_frames, output_path, frame_rate, depth_vid_scale):
    n_frames = depth_frames.shape[0]
    # set the last dim to zero, since the image is grayscale
    frame_size = depth_frames.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_vid = cv2.VideoWriter(filename=f'{output_path}.mp4',
                              fourcc=fourcc,
                              fps=frame_rate,
                              frameSize=frame_size,
                              isColor=False)

    # Loop through each frame of the matrix and write it to the video
    for i in range(n_frames):
        frame = depth_frames[i]
        im = frame * depth_vid_scale
        im = np.round(im).astype(np.uint8)
        out_vid.write(im)
    out_vid.release()  # Release the video
    print(f'Depth video saved to: {output_path}')


if __name__ == '__main__':
    main()
