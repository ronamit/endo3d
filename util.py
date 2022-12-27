import subprocess
import os


def frames2video(images_path, ffmpeg_path, seq_out_path, vid_file_name, frame_rate,
                 output_format='mp4'):
    '''
    Convert a sequence of images to a video using ffmpeg
    '''
    input_pattern = os.path.join(images_path, '*.png')
    output_file = os.path.join(seq_out_path, vid_file_name) + '.' + output_format
    codec = 'libx264'
    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-pattern_type', 'glob',
               '-i', input_pattern,
               '-vcodec', codec,
               output_file]
    subprocess.run(command)
