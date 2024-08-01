import plotly.graph_objects as go


def box(
    b: float,
    h: float,
    d: float,
    anchor=(0, 0, 0),
    color: str = "#aaaaaa",
    opacity: float = 0.5,
) -> go.Mesh3d:
    
    anchor_x, anchor_y, anchor_z = anchor

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
        x=x_array,
        y=y_array,
        z=z_array,
        opacity=opacity,
        color=color,
        alphahull=0
    )

    return mesh


def prism() -> go.Mesh3d:
    return go.Mesh3d()


def cone() -> go.Mesh3d:
    return go.Mesh3d()


