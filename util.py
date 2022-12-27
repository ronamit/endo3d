import subprocess
import os
import glob


def frames2video(images_path, ffmpeg_path, seq_out_path, vid_file_name, frame_rate,
                 output_format='mp4'):
    '''
    Convert a sequence of images to a video using ffmpeg
    '''

    # get the prefix of the image files
    image_prefix = glob.glob(os.path.join(images_path, '*.png'))[0][0:-9]

    output_file = os.path.join(seq_out_path, vid_file_name) + '.' + output_format

    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-i', image_prefix + '%05d.png',
               '-vcodec', 'libx265',  # codec
               '-crf', '15',   # compression vs. quality factor - see https://trac.ffmpeg.org/wiki/Encode/H.265
               output_file]
    subprocess.run(command)
    print(f'Video saved to: {output_file}')
