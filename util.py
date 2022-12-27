import subprocess
import os
import glob
import array

import numpy as np
import OpenEXR
import Imath
import cv2
import matplotlib.pyplot as plt


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

def create_depth_video(seq_in_path, seq_out_path, vid_file_name, frame_rate):
    '''
    Load a sequence of depth images from a folder
    '''
    # list of paths to EXR files
    exr_paths = glob.glob(os.path.join(seq_in_path, '*.exr'))
    exr_paths.sort()

    # Set the output video parameters
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # codec

    # Compute the size
    example_file = OpenEXR.InputFile(exr_paths[0])
    dw = example_file.header()['dataWindow']
    frame_size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    output_path = os.path.join(seq_out_path, vid_file_name) + '.exr'

    # Create the output video
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)

    # Iterate over the EXR files and add them to the video
    for exr_path in exr_paths:
        # Open the EXR file
        file = OpenEXR.InputFile(exr_path)

        # # Read channels as 32-bit floats
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (R, G, B, A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "A")]

        # the RGB channels are identical, and represent the depth
        depth_img = np.reshape(R, frame_size)

        # plt.imshow(depth_img, cmap='hot', interpolation='nearest')
        # plt.show()

        # Add the depth frame to the video
        out.write(depth_img)

    # Release the video writer
    out.release()
