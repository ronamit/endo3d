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


def get_frame_at_timestamp(dataset_path, seq_name, desired_time, vid_file_name):
    '''
    get the frame at a given timestamp
    :param desired_time:  the in-sequence time of the frame to load [seconds]]
    '''
    vid_path = os.path.join(dataset_path, seq_name, vid_file_name) + '.mp4'
    assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_interval = 1 / fps
    frame_no = 0
    frame_exists = False
    curr_frame = None
    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # [seconds]
            if desired_time - curr_time < t_interval / 2:
                break
        else:
            break
        frame_no += 1
    cap.release()
    frame_rgb = curr_frame[:, :, [2, 1, 0]]  # convert to RGB
    return frame_exists, frame_rgb
