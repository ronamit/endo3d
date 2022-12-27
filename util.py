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

    codec = 'libx264'
    command = [ffmpeg_path, '-y',
               '-framerate', str(frame_rate),
               '-i', image_prefix + '%05d.png',
               '-vcodec', codec,
               output_file]
    subprocess.run(command)
    # Note: '-i' is a workaround (https://stackoverflow.com/questions/31201164/ffmpeg-error-pattern-type-glob-was-selected-but-globbing-is-not-support-ed-by)
    print(f'Video saved to: {output_file}')