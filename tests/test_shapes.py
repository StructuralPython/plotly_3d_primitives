import numpy as np
import numpy.testing as npt
from plotly_3d_primitives import shapes


def test_build_point_matrix():
    x_array = np.array([0, 0])
    y_array = np.array([0, 1])
    z_array = np.array([0, 0])
    matrix = shapes.build_point_matrix(x_array, y_array, z_array)

    npt.assert_array_equal(
        matrix,
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, 0],
                [1, 1],
            ]
        ),
    )


def test_transform_points():
    x_array = np.array([0, 1])
    y_array = np.array([0, 0])
    z_array = np.array([0, 0])
    matrix = shapes.build_point_matrix(x_array, y_array, z_array)
    new_matrix_1 = shapes.transform_points(matrix, (0, 0, 0), (0, 1, 0))

    npt.assert_array_almost_equal(
        new_matrix_1,
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, 0],
            ]
        ),
    )

    new_matrix_2 = shapes.transform_points(matrix, (10, 10, 10), (0, 1, 0))
    npt.assert_array_almost_equal(
        new_matrix_2,
        np.array(
            [
                [10, 10],
                [10, 11],
                [10, 10],
            ]
        ),
    )
