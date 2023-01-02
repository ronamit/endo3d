import numpy as np


def depth_map_to_point_cloud(z_depth_frame, metadata):
    """
    """
    height, width, channels = z_depth_frame.shape
    n_pix = height * width

    cx = metadata['cx']  # middle of the image in x-axis [pixels]
    cy = metadata['cy']  # middle of the image in y-axis [pixels]
    sx = metadata['sx']   # the with of each pixel's sensor [millimeter/pixel]
    sy = metadata['sy']  # the height of each pixel's sensor [millimeter/pixel]
    focal_length = metadata['focal_length']  # [millimeter]

    # Get the pixels image coordinates (u, v) of the depth image [pixels]
    u_cords, v_cords = np.meshgrid(np.arange(0, width), np.arange(0, height))
    u_cords = u_cords.reshape((n_pix, 1))
    v_cords = v_cords.reshape((n_pix, 1))

    # get the corresponding coordinates in the camera system fo each pixel's sensor (the z axis is the optical
    # axis) [millimeter]
    camera_sys_cord = np.column_stack(((u_cords - cx) * sx,
                                       (v_cords - cy) * sy,
                                       focal_length * np.ones((n_pix, 1))))

    # normalize to get the unit vector from the origin (focal point) to the pixel's sensor, in the camera system
    pix_dir = camera_sys_cord / np.linalg.norm(camera_sys_cord, axis=1)[:, np.newaxis]

    # get the coordinate of the object surface each pixel is looking at, in the camera system
    surface_cord = camera_sys_cord + pix_dir * depth_frame.reshape((n_pix, 1))

    # reshape to the original image shape
    surface_cord = surface_cord.reshape((height, width, 1))

    # TODO: option get the RGB color of each pixel

    return surface_cord
