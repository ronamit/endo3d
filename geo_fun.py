import numpy as np


def depth_map_to_point_cloud(z_depth_frame, metadata):
    """
    """
    height, width, channels = z_depth_frame.shape
    n_pix = height * width

    cx = metadata['cx']  # middle of the image in x-axis [pixels]
    cy = metadata['cy']  # middle of the image in y-axis [pixels]
    sx = metadata['sx']  # the with of each pixel's sensor [millimeter/pixel]
    sy = metadata['sy']  # the height of each pixel's sensor [millimeter/pixel]
    focal_length = metadata['focal_length']  # [millimeter]

    # Get the pixels image coordinates (u, v) of the depth image [pixels]
    u_cords, v_cords = np.meshgrid(np.arange(0, width), np.arange(0, height))
    u_cords = u_cords.reshape((n_pix, 1))
    v_cords = v_cords.reshape((n_pix, 1))

    # get the corresponding coordinates in the camera system fo each pixel's sensor (the z axis is the optical
    # axis) [millimeter]
    pix_cord = np.column_stack(((u_cords - cx) * sx,
                                (v_cords - cy) * sy,
                                focal_length * np.ones((n_pix, 1))))

    # the surface point that each pixel is looking at is at a known z_depth,
    # and is on the ray connecting the focal point to the pixel's sensor.
    z_depth = z_depth_frame[:, :, 0].reshape((n_pix, 1))
    surface_cord = pix_cord * z_depth / focal_length

    # reshape to the original image shape (the last dim is the X,Y,Z  in the camera-system)
    surface_cord = surface_cord.reshape((height, width, 3))

    # TODO: option get the RGB color of each pixel

    return surface_cord
