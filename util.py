import glob
import os
import re
import numpy as np
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


def fig2img(fig):
    fig.canvas.draw()
    # convert canvas to image using numpy
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.flip(img, 1)
    return img


def get_frame_at_timestamp(seq_out_path, desired_time, vid_file_name):
    """
    get the frame at a given timestamp
    :param vid_file_name:
    :param seq_out_path:
    :param desired_time:  the in-sequence time of the frame to load [seconds]]
    """
    vid_path = os.path.join(seq_out_path, vid_file_name) + '.mp4'
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
    return frame_rgb, frame_exists


def depth_map_to_point_cloud(depth_frame, cam_K_mat):
    '''
    :param depth_frame: the depth frame
    :param depth_frame:
    :param cam_K_mat: The matrix of the camera intrinsics (camera_sys_hom_cord = cam_K_mat @ pixel_hom_cord)
    :return:
    '''
    height, width = depth_frame.shape
    n_pix = height * width

    # Get the pixels image coordinates (u, v) of the depth image
    u_cords, v_cords = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # create a matrix where each row is a pixel image hom. coordinate (u,v,1)
    pix_hom_cord = np.matrix([u_cords.flatten(), v_cords.flatten(), np.ones(n_pix)])

    # get the corresponding hom. coordinates in the camera system fo each pixel's sensor (the z axis is the optical
    # axis)
    camera_sys_hom_cord = np.linalg.solve(cam_K_mat, pix_hom_cord)
    # get the regular coordinates in the camera system
    camera_sys_cord = camera_sys_hom_cord[:, :3] / camera_sys_hom_cord[:, 4]

    # normalize to get the unit vector from the origin (focal point) to the pixel's sensor, in the camera system
    pix_dir = camera_sys_cord / np.linalg.norm(camera_sys_cord, axis=1)[:, np.newaxis]

    # get the coordinate of the surface each pixel is looking at, in the camera system (depth times the unit vector)
    surface_cord = pix_dir * depth_frame.flatten()

    # reshape to the original image shape
    surface_cord = surface_cord.reshape((height, width, 1))

    # TODO: option get the RGB color of each pixel

    return surface_cord

