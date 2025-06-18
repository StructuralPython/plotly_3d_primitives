import plotly.graph_objects as go
import numpy as np
import numpy.typing as nptp
from typing import Optional


def cube(
    center=(0.0, 0.0, 0.0),
    x_length=1.0,
    y_length=1.0,
    z_length=1.0,
    bounds: Optional[tuple[float]] = None,
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    if bounds is not None and len(bounds) == 6:
        x0, x1, y0, y1, z0, z1 = bounds
    else:
        x0 = center[0] - x_length / 2
        x1 = center[0] + x_length / 2

        y0 = center[1] - y_length / 2
        y1 = center[1] + y_length / 2

        z0 = center[2] - z_length / 2
        z1 = center[2] + z_length / 2

    x_array = [x0, x1, x1, x0, x0, x1, x1, x0]
    y_array = [y0, y0, y1, y1, y0, y0, y1, y1]
    z_array = [z0, z0, z0, z0, z1, z1, z1, z1]

    i_array = [0, 1, 4, 5, 0, 1, 0, 3, 3, 1, 1, 2]
    j_array = [1, 2, 5, 6, 1, 4, 3, 4, 2, 2, 5, 6]
    k_array = [3, 3, 7, 7, 4, 5, 4, 7, 7, 6, 6, 7]

    mesh = go.Mesh3d(
        x=x_array, y=y_array, z=z_array, opacity=opacity, color=color, alphahull=0
    )
    # mesh = go.Mesh3d(
    #     x=x_array,
    #     y=y_array,
    #     z=z_array,
    #     i=i_array,
    #     j=j_array,
    #     k=k_array,
    #     opacity=opacity,
    #     color=color,
    # )
    return mesh


def prism(
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    radius=0.5,
    height=1.0,
    n_sides: int = 4,
    capping=True,
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    anchor_x, anchor_y, anchor_z = center

    arr = np.linspace(0, 2 * np.pi, num=n_sides, endpoint=False)
    z_poly = np.cos(arr) * radius
    y_poly = np.sin(arr) * radius
    x_poly = np.zeros(n_sides)

    x_array = np.concatenate([x_poly, x_poly + height])
    y_array = np.concatenate([y_poly, y_poly])
    z_array = np.concatenate([z_poly, z_poly])

    x_array, y_array, z_array = apply_transformations(x_array, y_array, z_array, center, direction)

    return go.Mesh3d(
        x=x_array, y=y_array, z=z_array, alphahull=0, color=color, opacity=opacity
    )


def cone(
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    height=1.0,
    radius: Optional[float] = None,
    capping: bool = True,
    angle: float = 0.0,
    resolution: int = 6,
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    anchor_x, anchor_y, anchor_z = center

    arr = np.linspace(
        0 + np.radians(angle),
        2 * np.pi + np.radians(angle),
        num=resolution,
        endpoint=False,
    )
    z_poly = np.cos(arr) * radius
    y_poly = np.sin(arr) * radius
    x_poly = np.zeros(resolution)

    z_array = np.concatenate([z_poly, np.array([0])])
    y_array = np.concatenate([y_poly, np.array([0])])
    x_array = np.concatenate([x_poly, np.array([height])])

    x_array, y_array, z_array = apply_transformations(x_array, y_array, z_array, center, direction)

    return go.Mesh3d(
        x=x_array, y=y_array, z=z_array, alphahull=0, color=color, opacity=opacity
    )


def sphere(
    radius=0.5,
    center=(0.0, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    theta_resolution=30,
    phi_resolution=30,
    start_theta=0.0,
    end_theta=360.0,
    start_phi=0.0,
    end_phi=180.0,
    color="#555",
    opacity=0.5,
) -> go.Mesh3d:
    anchor_x, anchor_y, anchor_z = (0., 0., 0.)
    phi = np.linspace(start_phi, 2 * np.radians(end_phi), phi_resolution + 1)
    theta = np.linspace(start_theta, np.radians(end_theta), theta_resolution + 1)

    theta, phi = np.meshgrid(theta, phi)

    radius_zy = radius * np.sin(theta)
    z_array = np.ravel(anchor_z + np.cos(phi) * radius_zy)
    y_array = np.ravel(anchor_y + np.sin(phi) * radius_zy)
    x_array = np.ravel(anchor_x + radius * np.cos(theta))
    x_array, y_array, z_array = apply_transformations(x_array, y_array, z_array, center, direction)
    # print(center, direction)
    # print(y_array)
    return go.Mesh3d(
        x=x_array, y=y_array, z=z_array, alphahull=0, color=color, opacity=opacity
    )


def line(
    pointa=(-0.5, 0.0, 0.0),
    pointb=(0.5, 0.0, 0.0),
    resolution=1,
    color: str = "#aaa",
    line_width: float = 1.0,
    opacity: float = 0.8,
):
    """
    Returns a trace of a line in 3d space.
    """
    x0, y0, z0 = pointa
    x1, y1, z1 = pointb

    x_array = np.linspace(x0, x1, resolution + 1, endpoint=True)
    y_array = np.linspace(y0, y1, resolution + 1, endpoint=True)
    z_array = np.linspace(z0, z1, resolution + 1, endpoint=True)

    line_properties = dict(color=color, width=line_width)
    return go.Scatter3d(
        x=x_array,
        y=y_array,
        z=z_array,
        opacity=opacity,
        mode="lines",
        line=line_properties,
    )


def circular_arc_from_normal(
    center=(0.0, 0.0, 0.0),
    resolution=100,
    normal=[0.0, 0.0, 1.0],
    angle=90.0,
    polar=[1.0, 0.0, 0.0],
    color="#555",
    opacity=0.5,
    line_width=1.0,
    return_points=False,
) -> go.Mesh3d:
    """
    Create a circular arc defined by normal to the plane of the arc, and an
    angle.

    The number of segments composing the polyline is controlled by
    setting the object resolution.

    Parameters
    ----------
    center : sequence[float]
        Center of the circle that defines the arc.

    resolution : int, default: 100
        The number of segments of the polyline that draws the arc.
        Resolution of 1 will just create a line.

    normal : sequence[float], optional
        The normal vector to the plane of the arc.  By default it
        points in the positive Z direction.

    polar : sequence[float], optional
        Starting point of the arc in cartesian coordinates.  By default it
        is the unit vector in the positive x direction.

    angle : float, optional
        Arc length (in degrees) beginning at the polar vector.  The
        direction is counterclockwise.  By default it is 90.
    """
    center = np.array(center)
    anchor_x, anchor_y, anchor_z = center
    radius = np.sqrt(np.sum((polar - center) ** 2, axis=0))
    angles = np.linspace(0.0, np.radians(angle), resolution + 1, endpoint=True)

    z_array = np.cos(angles) * radius + anchor_z
    y_array = np.sin(angles) * radius + anchor_y
    x_array = np.zeros(resolution + 1) + anchor_x

    x_array, y_array, z_array = apply_transformations(x_array, y_array, z_array, center, normal)

    points = np.array([x_array, y_array, z_array]).T
    line_properties = dict(color=color, width=line_width)
    if return_points:
        return go.Scatter3d(
            x=x_array,
            y=y_array,
            z=z_array,
            line=line_properties,
            opacity=opacity,
            mode="lines",
        ), points
    else:
        return go.Scatter3d(
            x=x_array,
            y=y_array,
            z=z_array,
            line=line_properties,
            opacity=opacity,
            mode="lines",
        )


def rectangle(
    center=(0.0, 0.0, 0.0),
    b=1.0,
    d=1.0,
    normal=(1.0, 0.0, 0.0),
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    z0 = center[2] - b / 2
    z1 = center[2] + b / 2

    y0 = center[1] - d / 2
    y1 = center[1] + d / 2

    x0 = center[0]
    x1 = center[0]

    x_array = [x0, x1, x1, x0, x0, x1, x1, x0]
    y_array = [y0, y0, y1, y1, y0, y0, y1, y1]
    z_array = [z0, z0, z0, z0, z1, z1, z1, z1]

    x_array, y_array, z_array = apply_transformations(x_array, y_array, z_array, center, normal)

    mesh = go.Mesh3d(
        x=x_array,
        y=y_array,
        z=z_array,
        opacity=opacity,
        color=color,
        alphahull=0,
    )
    return mesh


def apply_transformations(
    x_array: nptp.ArrayLike,
    y_array: nptp.ArrayLike,
    z_array: nptp.ArrayLike,
    center: tuple,
    direction: tuple,
) -> nptp.ArrayLike:
    """
    Re-orients the point arrays to the new 'direction' and translates
    them to the new 'center'.
    """
    point_matrix = build_point_matrix(x_array, y_array, z_array)
    new_point_matrix = transform_points(point_matrix, center, direction)
    trans_x_array, trans_y_array, trans_z_array = new_point_matrix
    return trans_x_array, trans_y_array, trans_z_array


def build_point_matrix(
    x_array: nptp.ArrayLike,
    y_array: nptp.ArrayLike,
    z_array: nptp.ArrayLike,
) -> nptp.ArrayLike:
    """
    Returns a point matrix that can have transforms applied to it from a
    4x4 transformation matrix.
    """
    assert len(x_array) == len(y_array) == len(z_array)
    matrix = np.array([x_array, y_array, z_array, np.ones(len(x_array))])
    return matrix


def transform_points(
    point_matrix: nptp.ArrayLike,
    new_center: tuple,
    new_direction: tuple,
) -> nptp.ArrayLike:
    """
    Returns the 'point_matrix' transformed so that it's center is
    at 'new_center' and the point matrix is oriented toward
    'new_direction'. Function performs re-orientation then translation.

    point_matrix: a 4xn matrix where n is the number of points in
        the collection. All points must be in 3-space.
    new_center: a tuple indicating the center of the new point collection
        (which is originally centered on (0, 0, 0))
    new_direction: a tuple indicating a direction vector of the new point
        collection (which is originally oriented toward (1, 0, 0)).
    """
    transform_matrix = reorient(direction=new_direction)
    # print(transform_matrix)
    oriented_point_matrix = transform_matrix @ point_matrix
    oriented_point_matrix_3 = oriented_point_matrix[0:3]
    if not np.allclose(new_center, [0.0, 0.0, 0.0]):
        translated_point_matrix = (
            np.array(new_center, dtype=point_matrix.dtype) + oriented_point_matrix_3.T
        )
    else:
        translated_point_matrix = oriented_point_matrix_3.T
    return translated_point_matrix.T



# From Pyvista
def reorient(direction=(1.0, 0.0, 0.0)):
    """Create a transformation matrix to re-orient a point matrix
    to the provided direction.

    Parameters
    ----------
    direction : tuple, optional, default: (1.0, 0.0, 0.0)
        Direction vector along which the mesh should be oriented.

    """
    normx = np.array(direction) / np.linalg.norm(direction)
    normy_temp = [0.0, 1.0, 0.0]

    # Adjust normy if collinear with normx since cross-product will
    # be zero otherwise
    if np.allclose(normx, [0, 1, 0]):
        normy_temp = [-1.0, 0.0, 0.0]
    elif np.allclose(normx, [0, -1, 0]):
        normy_temp = [1.0, 0.0, 0.0]

    normz = np.cross(normx, normy_temp)
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    trans = np.zeros((4, 4))
    trans[:3, 0] = normx
    trans[:3, 1] = normy
    trans[:3, 2] = normz
    trans[3, 3] = 1

    return trans
