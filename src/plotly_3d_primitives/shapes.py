import plotly.graph_objects as go
import numpy as np


def box(
    b: float,
    d: float,
    h: float,
    align: list[str],
    anchor: tuple = (0, 0, 0),
    color: str = "#aaaaaa",
    opacity: float = 0.5,
) -> go.Mesh3d:

    anchor_point = anchor
    anchor_arr = np.array(anchor_point)
    for align_instr in align:
        if align_instr == "center":
            anchor_arr = anchor_arr - np.array([b / 2, d / 2, h / 2])

    anchor_x, anchor_y, anchor_z = anchor_arr

    x0 = anchor_x
    x1 = b + anchor_x

    y0 = anchor_y
    y1 = d + anchor_y

    z0 = anchor_z
    z1 = h + anchor_z

    x_array = [x0, x1, x1, x0, x0, x1, x1, x0]
    y_array = [y0, y0, y1, y1, y0, y0, y1, y1]
    z_array = [z0, z0, z0, z0, z1, z1, z1, z1]

    mesh = go.Mesh3d(
        x=x_array, y=y_array, z=z_array, opacity=opacity, color=color, alphahull=0
    )

    return mesh


def prism(
    n_points: int,
    h: float,
    align: list[str],
    anchor: tuple = (0, 0, 0),
    color: str = "#aaaaaa",
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
    color: str = "#aaaaaa",
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
