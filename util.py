import glob
import os

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def get_seq_id(seq_in_path):
    """
     get the 3 letter prefix of the files
    """
    seq_id = glob.glob(os.path.join(seq_in_path, '*.png'))[0][-13:-10]
    return seq_id


def fig2img(fig):
    fig.canvas.draw()
    # convert canvas to image using numpy
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.flip(img, 1)
    return img


def get_frame_at_timestamp(dataset_path, seq_name, frame_time, vid_file_name):
    '''
    get the frame at a given timestamp
    :param frame_time:  the in-sequence time of the frame to load [seconds]]
    '''
    vid_path = os.path.join(dataset_path, seq_name, vid_file_name) + '.mp4'
    assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_interval = 1 / fps
    frame_no = 0

    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        frame_time = frame_no * t_interval
        if frame_exists:
            print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
        else:
            break
        print("frame_no: ", frame_no)
        print("frame_time: ", frame_time)
        frame_no += 1

    cap.release()
    return

# def get_frame_at_timestamp(dataset_path, seq_name, frame_time, vid_file_name, ffmpeg_path):
#     '''
#     get the frame at a given timestamp
#     :param frame_time:  the in-sequence time of the frame to load [seconds]]
#     '''
#     vid_path = os.path.join(dataset_path, seq_name, vid_file_name) + '.mp4'
#     assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
#
#     frame_rate = 20
#
#     command = [ffmpeg_path,
#                '-i', vid_path,
#                '-f', 'image2pipe',
#                '-pix_fmt', 'rgb24',
#                '-r', str(frame_rate),  # frames per second
#                # '-ss', str(frame_time),  # ss: seek to given time position
#                'temp.png'] # output file
#     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
#     raw_image = pipe.stdout.read(512 * 512 * 3)
#     return


# def get_frame_at_timestamp(dataset_path, seq_name, frame_time, vid_file_name, ffmpeg_path):
#     '''
#     get the frame at a given timestamp
#     :param frame_time:  the in-sequence time of the frame to load [seconds]]
#     '''
#     vid_path = os.path.join(dataset_path, seq_name, vid_file_name) + '.mp4'
#     assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
#
#     frame_rate = 20
#
#     vidin = ffmpegcv.VideoCapture(vid_path)
#     w, h = vidin.width, vidin.height
#
#     for frame in vidin:
#         pass
#
#     vidin.release()
#     return
#
