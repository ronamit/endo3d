import glob
import os
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_seq_id(seq_in_path):
    """
     get the 3 letter prefix of the files
    """
    seq_id = glob.glob(os.path.join(seq_in_path, '*.png'))[0][-13:-10]
    return seq_id


def get_frame_at_timestamp(dataset_path, seq_name, frame_time, vid_file_name):
    '''
    get the frame at a given timestamp
    :param dataset_path:
    :type dataset_path:
    :param seq_name:
    :type seq_name:
    :param frame_time:  the in-sequence time of the frame to load [seconds]
    :type frame_time:
    :param vid_file_name:
    :type vid_file_name:
    :return:
    :rtype:
    '''
    vid_path = os.path.join(dataset_path, seq_name, vid_file_name) + '.mp4'
    assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    t_interval = 1 / fps
    frame_no = 0
    while vidcap.isOpened():
        frame_exists, curr_frame = vidcap.read()
        frame_time = frame_no * t_interval
        if frame_exists:
            print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
        else:
            break
        print("frame_no: ", frame_no)
        print("frame_time: ", frame_time)
        frame_no += 1

    vidcap.release()
    return
