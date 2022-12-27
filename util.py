import subprocess
import os

def frames2video(images_path, ffmpeg_path, dataset_out_path, vid_file_name, frame_rate,
                 output_format='mp4'):
    '''
    Convert a sequence of images to a video using ffmpeg
    :param output_file_name:
    :param output_dir_path:
    :param output_format:
    :param images_path: path to the input images
    :param ffmpeg_path:
    :param output_path: path to the output video
    :param frame_rate: frame rate
    :return:
    '''
    input_pattern = os.path.join(images_path, '*.png')
    output_file = os.path.join(dataset_out_path, vid_file_name) + '.' + output_format
    codec = 'libx264'
    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-pattern_type', 'glob',
               '-i', input_pattern,
               '-vcodec', codec,
               output_file]
    subprocess.run(command)
