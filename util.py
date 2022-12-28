import subprocess
import os
import glob
import array

import numpy as np
import OpenEXR
import Imath
import cv2
import matplotlib.pyplot as plt
import h5py


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


# def create_depth_video(seq_in_path, seq_out_path, vid_file_name, frame_rate):
#     '''
#     Load a sequence of depth images from a folder
#     '''
#     # list of paths to EXR files
#     exr_paths = glob.glob(os.path.join(seq_in_path, '*.exr'))
#     exr_paths.sort()
#
#
#     # Open the output file for writing
#     output_path = os.path.join(seq_out_path, vid_file_name) + '.exr'
#     out_file = OpenEXR.OutputFile(output_path)
#
#     # Loop through each EXR file and write the depth data to the output file
#     for i, exr_path in enumerate(exr_paths):
#         # Open the EXR file and read the depth data
#         exr_file = OpenEXR.InputFile(exr_path)
#         depth_data = np.array(exr_file.channel(0, Imath.PixelType(Imath.PixelType.FLOAT)))
#
#         # Write the depth data to the output file
#         out_file.writePixels({'depth': depth_data})
#
#     # Close the output file
#     out_file.close()

def save_depth_frames(seq_in_path, seq_out_path, vid_file_name, frame_rate):
    '''
    Load a sequence of depth images from a folder
    '''
    # list of paths to EXR files
    exr_paths = glob.glob(os.path.join(seq_in_path, '*.exr'))

    # exr_paths = exr_paths[:20] # debug

    exr_paths.sort()
    n_frames = len(exr_paths)

    # Open the output file for writing
    output_path = os.path.join(seq_out_path, vid_file_name) + '.h5'

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
        # plt.imshow(depth_frames[i_frame], cmap='hot', interpolation='nearest')
        # plt.show()
        # break

    print(f'Min depth: {np.min(depth_frames.flatten())}, Max depth: {np.max(depth_frames.flatten())}')
    plt.hist(depth_frames.flatten(), bins='auto')
    plt.show()

    # Save
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset(vid_file_name, data=depth_frames)
    print(f'Depth frames saved to: {output_path}')

    #     # initialize video writer
    #     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
    #                           fps=frame_rate,
    #                           frameSize=(frame_size[1], frame_size[0]),
    #                           isColor=False)
    #
    #     # add this array to the video
    #     out.write(depth_img)
    #
    # # close out the video writer
    # out.release()
