import array
import glob
import os
import subprocess

import Imath
import OpenEXR
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


def get_seq_id(seq_in_path):
    """
     get the 3 letter prefix of the files
    """
    seq_id = glob.glob(os.path.join(seq_in_path, '*.png'))[0][-13:-10]
    return seq_id


def create_rgb_video(seq_in_path, seq_out_path, vid_file_name, frame_rate, ffmpeg_path):
    """
    Convert a sequence of images to a video using ffmpeg
    """

    output_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'
    seq_id = get_seq_id(seq_in_path)
    frames_paths = glob.glob(os.path.join(seq_in_path, f'{seq_id}_*.png'))
    image_prefix = os.path.join(seq_in_path, seq_id + '_')
    print(f'Number of RGB frames to be loaded: {len(frames_paths)}')

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


def save_depth_video(depth_frames, output_path, frame_rate, mode='heatmap'):
    n_frames = depth_frames.shape[0]
    frame_size = depth_frames.shape[1:]
    fcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # codec
    out = cv2.VideoWriter(f'{output_path}_video_{mode}.mp4',
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
        # redraw the canvas
        fig = plt.gcf()
        fig.canvas.draw()
        # convert canvas to image using numpy
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np.flip(img_cv, 1)
        out.write(img_cv)
        plt.cla()
    plt.close(fig)
    # Release the VideoWriter object
    out.release()
    print(f'Depth video saved to: {output_path}')


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
    print(f'Number of depth frames to be loaded: {n_frames}')

    # Open the output file for writing
    output_path = os.path.join(seq_out_path, vid_file_name)

    # Compute the size
    example_file = OpenEXR.InputFile(depth_files_paths[0])
    dw = example_file.header()['dataWindow']
    frame_size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    depth_frames = np.zeros((n_frames, frame_size[0], frame_size[1]), dtype=np.float32)

    # Iterate over the EXR files and add them to the video
    for i_frame, exr_path in enumerate(depth_files_paths):
        # Open the EXR file
        file = OpenEXR.InputFile(exr_path)
        # Read channels as floats, the RGB channels are identical, and represent the depth
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        R_chan = array.array('f', file.channel('R', FLOAT)).tolist()
        depth_img = np.reshape(R_chan, frame_size)
        depth_frames[i_frame] = depth_img

    print(f'Min depth: {np.min(depth_frames.flatten())}, Max depth: {np.max(depth_frames.flatten())}')
    plt.hist(depth_frames.flatten(), bins='auto')
    plt.show()
    plt.savefig(output_path + '_histogram.png')

    # Save as matrix
    with h5py.File(output_path + '.h5', 'w') as hf:
        hf.create_dataset(vid_file_name, data=depth_frames, compression='gzip')
    print(f'Depth frames saved to: {output_path}.h5')

    save_depth_video(depth_frames, output_path, frame_rate, mode='heatmap')
