import numpy as np


def depth_map_to_point_cloud(depth_frame, cam_K_mat):
    '''
    :param depth_frame: the depth frame
    :param depth_frame:
    :param cam_K_mat: The matrix of the camera intrinsics (camera_sys_hom_cord = cam_K_mat @ pixel_hom_cord)
    :return:
    '''
    height, width, channels = depth_frame.shape
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
