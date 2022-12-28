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


def create_rgb_video(seq_in_path, seq_out_path, vid_file_name, frame_rate, ffmpeg_path):
    '''
    Convert a sequence of images to a video using ffmpeg
    '''

    # get the prefix of the image files
    image_prefix = glob.glob(os.path.join(seq_in_path, '*.png'))[0][0:-9]

    output_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'

    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-i', image_prefix + '%05d.png',
               '-vcodec', 'libx265',  # codec
               '-crf', '15',  # compression vs. quality factor - see https://trac.ffmpeg.org/wiki/Encode/H.265
               output_path]
    subprocess.run(command)
    print(f'Video saved to: {output_path}')


def save_heatmap_video(depth_frames, output_path, frame_rate):
    n_frames = depth_frames.shape[0]
    frame_size = depth_frames.shape[1:]
    # Set the video codec
    fcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path + '_video.mp4',
                          fcc,
                          fps=frame_rate,
                          frameSize=(frame_size[1], frame_size[0]),
                          isColor=False)

    # Loop through each frame of the matrix and write it to the video
    for i in range(n_frames):
        frame = depth_frames[i]
        fig, ax = plt.subplots()
        dpi = 100
        fig.set_size_inches(frame_size[0] / dpi, frame_size[1] / dpi)
        ax.imshow(frame, cmap='hot', interpolation='nearest')
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
        imf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(imf)

    # Release the VideoWriter object
    out.release()
    print(f'Depth video saved to: {output_path}')


def save_depth_frames(seq_in_path, seq_out_path, vid_file_name, frame_rate):
    '''
    Load a sequence of depth images from a folder
    '''
    # list of paths to EXR files
    exr_paths = glob.glob(os.path.join(seq_in_path, '*.exr'))

    exr_paths = exr_paths[:20]  # debug

    exr_paths.sort()
    n_frames = len(exr_paths)

    # Open the output file for writing
    output_path = os.path.join(seq_out_path, vid_file_name)

    # Compute the size
    example_file = OpenEXR.InputFile(exr_paths[0])
    dw = example_file.header()['dataWindow']
    frame_size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    depth_frames = np.zeros((n_frames, frame_size[0], frame_size[1]), dtype=np.float32)

    # Iterate over the EXR files and add them to the video
    for i_frame, exr_path in enumerate(exr_paths):
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

    # Save as matrix
    with h5py.File(output_path + '.h5', 'w') as hf:
        hf.create_dataset(vid_file_name, data=depth_frames)
    print(f'Depth frames saved to: {output_path}.h5')

    save_heatmap_video(depth_frames, output_path, frame_rate)
