'''

'''

import argparse
import glob
import os
import shutil
import subprocess

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt

from util import get_seq_id, fig2img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg_path', type=str, required=True, help='path to ffmpeg executable')
    parser.add_argument('--sim_out_path', type=str, required=True,
                        help=' path to the Unity simulator output')
    args = parser.parse_args()

    sim_out_path = os.path.abspath(args.sim_out_path)
    dataset_out_path = sim_out_path + '_Dataset'
    if not os.path.isdir(dataset_out_path):
        os.makedirs(dataset_out_path)
    print(f'Datasets will be saved to {dataset_out_path}')
    seq_paths = glob.glob(sim_out_path + '/*/')
    print('Simulation sequences to be loaded: ', seq_paths)

    for seq_in_path in seq_paths:
        print('Loading: ', seq_in_path)
        seq_name = os.path.basename(os.path.normpath(seq_in_path))
        seq_out_path = os.path.join(dataset_out_path, seq_name)
        if not os.path.isdir(seq_out_path):
            os.makedirs(seq_out_path)
        shutil.copy2(os.path.join(seq_in_path, 'MySettings.set'),
                     os.path.join(seq_out_path, seq_name + '_Settings.set'))

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
                          vid_file_name=seq_name + '_Depth',
                          frame_rate=frame_rate)

        create_rgb_video(seq_in_path=seq_in_path,
                         seq_out_path=seq_out_path,
                         vid_file_name=seq_name + '_RGB',
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


def create_rgb_video(seq_in_path, seq_out_path, vid_file_name, frame_rate, ffmpeg_path):
    """
    Convert a sequence of images to a video using ffmpeg
    """

    output_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'
    seq_id = get_seq_id(seq_in_path)
    frames_paths = glob.glob(os.path.join(seq_in_path, f'{seq_id}_*.png'))
    image_prefix = os.path.join(seq_in_path, seq_id + '_')
    print(f'Number of RGB frames: {len(frames_paths)}')

    command = [ffmpeg_path,
               '-hide_banner', '-loglevel', 'error', '-nostats',  # less verbose
               '-y',
               '-framerate', str(frame_rate),
               '-i', image_prefix + '%05d.png',
               '-vcodec', 'libx265',  # codec
               '-crf', '15',  # compression vs. quality factor - see https://trac.ffmpeg.org/wiki/Encode/H.265,
               output_path]
    subprocess.run(command)
    print(f'Video saved to: {output_path}')


def save_depth_frames(seq_in_path, seq_out_path, vid_file_name, frame_rate, limit_frame_num=0):
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
        # All 3 channels are the same (depth)
        depth_img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

        # R_chan = array.array('f', file.channel('R', FLOAT)).tolist()
        # depth_img = np.reshape(R_chan, frame_size)
        depth_frames[i_frame] = depth_img

    print(f'Min depth: {np.min(depth_frames.flatten())}, Max depth: {np.max(depth_frames.flatten())}')
    plt.hist(depth_frames.flatten(), bins='auto')
    plt.savefig(output_path + '_histogram.png')
    plt.close()

    # Save as matrix
    with h5py.File(output_path + '.h5', 'w') as hf:
        hf.create_dataset(vid_file_name, data=depth_frames, compression='gzip')
    print(f'Depth frames saved to: {output_path}.h5')

    save_depth_video(depth_frames, output_path, frame_rate, mode='heatmap')


def save_depth_video(depth_frames, output_path, frame_rate, mode='heatmap'):
    n_frames = depth_frames.shape[0]
    frame_size = depth_frames.shape[1:]
    fcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # codec
    out = cv2.VideoWriter(f'{output_path}_{mode}.mp4',
                          fcc,
                          fps=frame_rate,
                          frameSize=(frame_size[1], frame_size[0]))

    fig, ax = plt.subplots()
    # Loop through each frame of the matrix and write it to the video
    for i in range(n_frames):
        frame = depth_frames[i]
        dpi = 100
        fig.set_size_inches(frame_size[0] / dpi, frame_size[1] / dpi)
        if mode == 'heatmap':
            ax.imshow(frame, cmap='hot', interpolation='nearest')
        elif mode == 'validity':
            ax.imshow(frame > 0, interpolation='nearest')  # a binary image
        else:
            raise ValueError
        plt.axis('off')
        plt.axis('image')
        # remove white padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im = fig2img(fig)
        out.write(im)
        plt.cla()
    plt.close(fig)
    # Release the VideoWriter object
    out.release()
    print(f'Depth video saved to: {output_path}')


if __name__ == '__main__':
    main()
