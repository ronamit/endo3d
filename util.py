import subprocess
import os
import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def frames2video(images_path, ffmpeg_path, output_dir_path='', output_file_name='video', frame_rate=30,
                 output_format='mp4'):
    '''
    Convert a sequence of images to a video using ffmpeg
    :param output_format:
    :param images_path: path to the input images
    :param ffmpeg_path:
    :param output_path: path to the output video
    :param frame_rate: frame rate
    :return:
    '''
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    input_pattern = os.path.join(images_path, '*.png')
    output_file = os.path.join(output_dir_path, output_file_name) + '.' + output_format
    codec = 'libx264'
    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-pattern_type', 'glob',
               '-i', input_pattern,
               '-vcodec', codec,
               output_file]
    print(command)
    subprocess.run(command)
