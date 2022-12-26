import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def frames2video(dir_path, output_path='', framerate=30):

    # Set the input and output file names
    input_pattern = os.path.join(dir_path, '%s_%d.png')
    output_file = output_path + '.mp4'

    # Set the FFmpeg command
    command = ['ffmpeg', '-y',  # Overwrite output file if it exists
               '-framerate', f'{framerate}',  # Set the frame rate
               '-i', input_pattern,  # Set the input file pattern
               '-c:v', 'libx264',  # Set the video codec
               '-pix_fmt', 'yuv420p',  # Set the pixel format
               output_file]  # Set the output file name
    subprocess.run(command)
