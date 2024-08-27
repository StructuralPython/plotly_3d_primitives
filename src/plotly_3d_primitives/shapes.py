import plotly.graph_objects as go
import numpy as np
from typing import Optional


def cube(
    center=(0.0, 0.0, 0.0),
    x_length=1.0,
    y_length=1.0,
    z_length=1.0,
    bounds: Optional[tuple[float]] = None,
    color: str = "#aaaaaa",
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
