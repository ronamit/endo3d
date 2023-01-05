import numpy as np
import plotly.graph_objects as go


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
    surface_cord = sensor_cord * z_depth / focal_length  # the X,Y,Z  in the camera-system

    # TODO: option get the RGB color of each pixel

    return surface_cord, sensor_cord


def z_depth_map_to_ray_depth_map(z_depth_frame, metadata):
    height, width = z_depth_frame.shape
    n_pix = height * width
    surface_cord, sensor_cord = z_depth_map_to_point_cloud(z_depth_frame, metadata)
    surface_cord = surface_cord.reshape((n_pix, 3))
    ray_depth_map = np.linalg.norm(surface_cord - sensor_cord, axis=-1)
    ray_depth_map = ray_depth_map.reshape((height, width))
    return ray_depth_map


# def plot_3d_point_cloud(point_cloud, title=''):
#     """
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     ax.set_title(title)
#     plt.show()
#

def plot_3d_point_cloud(surface_point, sensor_points, color_by='Ray_Depth'):
    n_samples = surface_point.shape[0]
    ray_depth = np.linalg.norm(surface_point - sensor_points, axis=-1)

    if color_by == 'Ray_Depth':
        # color the surface point according to ray depth, and set the sensor point to black
        colors = np.concatenate((ray_depth, np.zeros(n_samples)))
    else:
        raise ValueError('Unknown color_by option')
    # elif color_by == 'RGB':
    #     # color the surface point according to RGB, and set the sensor point to black
    #     colors = [rgb_frame[:, i_chan].tolist() + [0 for _ in range(n_samples)]
    #               for i_chan in range(3)]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=surface_point[:, 0],
            y=surface_point[:, 1],
            z=surface_point[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8))])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(
                          xaxis_title='X [mm]',
                          yaxis_title='Y [mm]',
                          zaxis_title='Z [mm]'))

    # add the image plane
    fig.add_trace(
            go.Mesh3d(
                x=[sensor_points[:, 0].min(), sensor_points[:, 0].max(), sensor_points[:, 0].max(), sensor_points[:, 0].min()],
                y=[sensor_points[:, 1].min(), sensor_points[:, 1].min(), sensor_points[:, 1].max(), sensor_points[:, 1].max()],
                z=[sensor_points[:, 2].min(), sensor_points[:, 2].min(), sensor_points[:, 2].min(), sensor_points[:, 2].min()],
                opacity=0.6)
    )
    fig.show()
