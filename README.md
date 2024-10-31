# plotly_3d_primitives

A small library for quickly generating 3d traces (`Mesh3d` and `Scatter3d`) of primitive shapes.

The API is intentionally designed to mimic PyVista so that, for the supported geometries, it could be a drop-in replacement.

## Primitives currently supported

- `cube`
- `sphere`
- `prism` (with n-sides)
- `cone`
- `line`
- `circular_arc_from_normal`

## Installation and basic usage

```
pip install plotly_3d_primitives
```

Each primitive is created from a function which returns a `plotly.graph_objects.Trace`, generally either a `plotly.graph_objects.Mesh3d` or a `plotly.graph_objects.Scatter3d`.

```python
import plotly_3d_primitives as prims
import plotly.graph_objects as go

a_cube = prims.cube(x_length=30, y_length=24, z_length=18, color='teal', opacity=0.5)

a_sphere = prims.sphere(radius=6.8, center=(-3, 0, 20), color="papayawhip", opacity=0.8) # everyone's favourite colour

a_cylinder = prims.prism(radius=10, center=(4, 10, 14), height=12, color="goldenrod", n_sides=12)

an_arc = prims.circular_arc_from_normal(center= (20, 10, 10), normal=(0, 1, 0), angle=215, color='red', line_width=10)

a_cone = prims.cone(center=(-5, -10, -6), direction=(0, 0, 1), height=14, radius=4, color='green', opacity=0.2)

a_line = prims.line(pointa=(1, -5, -10), pointb=(14, 23, 10), color="#333", opacity=1.0, line_width=15)

fig = go.Figure()
layout = go.Layout(scene=dict(aspectmode='data'), width=800, height=800)
fig.layout = layout

fig.add_traces([a_cube, a_sphere, a_cylinder, an_arc, a_cone, a_line])
fig.show()
```

## Motivation

Getting easy-and-consistent 3d geometry display in Python for the web can be fussy. VTK and PyVista are great buuuuuttt...PyVista (currently) does not work with streamlit. PyVista also requires a LOT of dependencies to be installed in order to work in a Jupyter environment.

Plotly is easy to install and displays in any web environment. It has great built-in display features (hover, annotations, click-button screenshots) and an easy-to-understand data model.

So, I have implemented **some** of PyVista's functionality in Plotly so it can be used as a drop-in replacement for these primitives.

**Contributions for other primitives and additional PyVista functionality is welcome.**

The functionality currently implemented is enough for me to create an alternative visualization module for the structural analysis library, [PyNite](https://github.com/jwock82/pynite). This was the primary motivation for creating this library but I have also used it in other locations.

**Note:** Plotly is not designed to handle large amounts of detailed 3d meshes. If you use this library to build large models, you can stall-out plotly and hang your browser. So, try to keep it light.
