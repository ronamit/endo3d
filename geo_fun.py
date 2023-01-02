import numpy as np
from matplotlib import pyplot as plt


def z_depth_map_to_point_cloud(z_depth_frame, metadata):
    """
    """
    height, width = z_depth_frame.shape
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
    sensor_cord = np.column_stack(((u_cords - cx) * sx,
                                   (v_cords - cy) * sy,
                                   focal_length * np.ones((n_pix, 1))))

    # the surface point that each pixel is looking at is at a known z_depth,
    # and is on the ray connecting the focal point to the pixel's sensor.
    z_depth = z_depth_frame.reshape((n_pix, 1))
    point_cloud = sensor_cord * z_depth / focal_length  # the X,Y,Z  in the camera-system

    # TODO: option get the RGB color of each pixel

    return point_cloud, sensor_cord


def z_depth_map_to_ray_depth_map(z_depth_frame, metadata):
    height, width = z_depth_frame.shape
    n_pix = height * width
    surface_cord, sensor_cord = z_depth_map_to_point_cloud(z_depth_frame, metadata)
    surface_cord = surface_cord.reshape((n_pix, 3))
    ray_depth_map = np.linalg.norm(surface_cord - sensor_cord, axis=-1)
    ray_depth_map = ray_depth_map.reshape((height, width))
    return ray_depth_map


def plot_3d_point_cloud(point_cloud, title=''):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    plt.show()
