import plotly.graph_objects as go
import numpy as np
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
    n_points: int,
    h: float,
    align: list[str],
    anchor: tuple = (0, 0, 0),
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    anchor_x, anchor_y, anchor_z = anchor

    arr = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
    x_poly = np.cos(arr) + anchor_x
    y_poly = np.sin(arr) + anchor_y
    z_poly = np.zeros(n_points) + anchor_z

    x_array = np.concat([x_poly, x_poly])
    y_array = np.concat([y_poly, y_poly])
    z_array = np.concat([z_poly, z_poly + h])

    return go.Mesh3d(
        x=x_array, y=y_array, z=z_array, alphahull=0, color=color, opacity=opacity
    )


def cone(
    n_points: int,
    h: float,
    align: list[str],
    anchor: tuple = (0, 0, 0),
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    anchor_x, anchor_y, anchor_z = anchor

    arr = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
    x_poly = np.cos(arr) + anchor_x
    y_poly = np.sin(arr) + anchor_y
    z_poly = np.zeros(n_points) + anchor_z

    x_array = np.concat([x_poly, np.array([anchor_x])])
    y_array = np.concat([y_poly, np.array([anchor_y])])
    z_array = np.concat([z_poly, np.array([anchor_z + h])])

    return go.Mesh3d(
        x=x_array, y=y_array, z=z_array, alphahull=0, color=color, opacity=opacity
    )


def line(
    pointa=(-0.5, 0.0, 0.0),
    pointb=(0.5, 0.0, 0.0),
    resolution=1,
    color: str = "#aaa",
    opacity: float = 0.8
):
    """
    Returns a trace of a line in 3d space.
    """
    x0, y0, z0 = pointa
    x1, y1, z1 = pointb

    x_array = np.linspace(x0, x1, resolution + 1, endpoint=True)
    y_array = np.linspace(y0, y1, resolution + 1, endpoint=True)
    z_array = np.linspace(z0, z1, resolution + 1, endpoint=True)

    return go.Scatter3d(x=x_array, y=y_array, z=z_array, color=color, opacity=opacity, mode='lines')


def rectangle(
    center=(0.0, 0.0, 0.0),
    b=1.0,
    d=1.0,
    normal=(1.0, 0.0, 0.0),
    color: str = "#aaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    x0 = center[0] - b / 2
    x1 = center[0] + b / 2

    y0 = center[1] - d / 2
    y1 = center[1] + d / 2

    z0 = center[2] - z_length / 2
    z1 = center[2] + z_length / 2

    x_array = [x0, x1, x1, x0, x0, x1, x1, x0]
    y_array = [y0, y0, y1, y1, y0, y0, y1, y1]
    z_array = [z0, z0, z0, z0, z1, z1, z1, z1]

    i_array = [0, 1]
    j_array = [1, 2]
    k_array = [3, 3]

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


def rectangular_grid(
    center=(0.0, 0.0, 0.0),
    b=1.0,
    d=1.0,
    normal=(0., 0., 1.0),
    rows=1,
    cols=1,
    color: str = "#aaa"
) -> go.Mesh3d:
    """
    Returns a grid like:
     ... ... ...
    | . | . | . |
    | 3 | 4 | 5 | ...
    | 0 | 1 | 2 | ...

    Where 0, 1, 2, 3, 4, ... etc. are the "indexes" of the grid
    rectangles. They are numbered from the bottom-left left-to-right,
    down-to-up until the bth rectangle which has an index of (m * n - 1)
    where m is rows and n is columns.

    b: total width of grid
    d: total depth (height) of grid

    color: str | dict[Callable, str] will color the rectangles either all
        one color (str) or conditionally color them based on whether the
        index value of each rectangle returns True in the dict callable key
        (the color in the value will be applied if True; the first matching
        condition applies).

    """
    # nodes
    center = np.array(center)
    normal = np.array(normal)

    # Direction cosines
    x_dir = (1, 0, 0)
    y_dir = (0, 1, 0)
    z_dir = (0, 0, 1)
    assumed_normal = z_dir
    norm = np.linalg.norm(normal)
    cos_alpha = np.dot(x_dir, normal) / norm
    cos_beta = np.dot(y_dir, normal) / norm
    cos_gamma = np.dot(z_dir, normal) / norm

    pitch_rot = np.array([
        [cos_beta, 0, 1-cos_beta],
        [0, 1, 0],
        [-1-cos_beta, 0, cos_beta]
    ])

    yaw_rot = np.array([
        [cos_gamma, -1-cos_gamma, 0],
        [1-cos_gamma, cos_gamma, 0],
        [0, 0, 1]
    ])


    # # The projection of the normal on each plane
    # xy_proj = np.array([normal[0], normal[1], 0])
    # yz_proj = np.array([0, normal[1], normal[2]])
    # xz_proj = np.array([normal[0], 0, normal[2]])

    # xy_perp = np.array([-normal[1], normal[0], 0])
    # yz_perp = np.array([0, normal[2], -normal[1]])
    # xz_perp = np.array([normal[2], 0, -normal[0]])

    # # Go "down" to the mid-point of the bottom edge of the rectangle
    # bot_mid_point = (
    #     center 
    #     - (
    #         yz_perp / np.linalg.norm(yz_perp) if np.sum(yz_perp) != 0 else np.zeros(3)
    #         + xz_perp / np.linalg.norm(xz_perp) if np.sum(xz_perp) != 0 else np.zeros(3)
    #     ) / 2**0.5 * d/2
    # )
    # # Then go "left" to the bottom-left corner of the rectangle for the "A" point
    # A_point = bot_mid_point - (xy_perp / np.linalg.norm(xy_perp) if np.sum(xy_perp) != 0 else np.zeros(3)) * b/2

    # # Go "up" to the mid-poitn of the top edge of the rectangle
    # top_mid_point = (
    #     center 
    #     + (
    #         yz_perp / np.linalg.norm(yz_perp) if np.sum(yz_perp) != 0 else np.zeros(3)
    #         + xz_perp / np.linalg.norm(xz_perp) if np.sum(xz_perp) != 0 else np.zeros(3)
    #     ) / 2**0.5 * d/2
    # )
    # print(bot_mid_point, top_mid_point)
    # # Then go "right" to the top-right corner of the rectangle for the "B" point
    # B_point = top_mid_point + (xy_perp / np.linalg.norm(xy_perp) if np.sum(xy_perp) != 0 else np.zeros(3)) * b/2
    # hypot_length = np.sqrt((b)**2 + (d)**2)

    # print(hypot_length)
    # print(np.linalg.norm(B_point - A_point))


    # Plane equation
    "normal[0] * x + normal[1] * y + normal[2] * z = np.dot(center, normal)"

    # Distance from center to "min" point
    "sqrt((b/2 - center[0])**2 + (d/2 - center[1])**2 + (0 - center[2])**2)"

    # A is "min" point, B is "max" point
    "tan(alpha) = d / b"

    # Assumption, the bottom edge of the rect will be parallel with the xy plane
    # Therefore vector of the bottom edge will be the -ve reciprocal of the xy projection of the vector normal [1, 2, 3] => [1, 2, 0]
    # So the bottom edge will be [-2, 1, 0]


    # triangles
    for j in range(rows):
        mod_co = cols + 1
        for i in range(cols):
            mod_ro = rows + 1
            rect_index = i + j * mod_co
            anchor_node = rect_index + j

            tri_1 = [anchor_node, anchor_node + mod_ro, anchor_node + mod_ro + 1]
            tri_2 = [anchor_node, anchor_node + 1, anchor_node + mod_ro + 1]

