import glob
import os
import re
import json

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def get_seq_id(seq_in_path):
    """
     get the 3 letter prefix of the files
    """
    seq_id = glob.glob(os.path.join(seq_in_path, '*.png'))[0][-13:-10]
    return seq_id


def find_between_str(file_path, before_str, after_str):
    """
    find the string between two strings in a file
    :param file_path: the path to the file
    :param before_str: the string before the desired string
    :param after_str: the string after the desired string
    :return: the desired string
    """
    matches = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                match = re.search(f"(?<={before_str}).*?(?={after_str})", line).group(0)
                matches.append(match)
            except AttributeError:
                pass
    assert len(matches) == 1, f'Found {len(matches)} matches for {before_str} and {after_str} in {file_path}'
    return matches[0]


def get_metadata(dataset_path, seq_name):
    seq_out_path = os.path.join(dataset_path, seq_name)
    metadata_path = os.path.join(seq_out_path, seq_name + '_metadata.json')
    with open(metadata_path, 'r') as fp:
        metadata = json.load(fp)
    return metadata


def get_frame_at_timestamp(seq_out_path, desired_time, vid_file_name, frame_rate, is_grayscale=False):
    """
    get the frame at a given timestamp
    :param frame_rate:
    :param vid_file_name:
    :param seq_out_path:
    :param desired_time:  the in-sequence time of the frame to load [seconds]]
    """
    vid_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'
    assert os.path.isfile(vid_path), f'Video file not found: {vid_path}'
    cap = cv2.VideoCapture(vid_path)
    t_interval = 1 / frame_rate
    frame_no = 0
    frame_exists = False
    curr_frame = None
    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        curr_time = frame_no * t_interval
        if frame_exists:
            if desired_time - curr_time < t_interval / 2:
                break
        else:
            break
        frame_no += 1
    cap.release()
    frame_rgb = curr_frame[:, :, [2, 1, 0]]  # convert to RGB
    return frame_rgb, frame_exists

