"""Test sparse difference matrix utilities."""

from numpy import array
from numpy.testing import assert_array_equal

from pmlib.utils.matrices import second_difference_matrix


def test_second_difference_matrix_coefficients() -> None:
    """
    Assert that the second difference matrix contains the expected
    coefficients at all relevant indices.
    """
    matrix = second_difference_matrix()

    assert_array_equal(array([-2, -2, -2]), matrix.A.diagonal(0))
    assert_array_equal(array([1, 1]), matrix.A.diagonal(1))
    assert_array_equal(array([1, 1]), matrix.A.diagonal(-1))


def test_second_difference_matrix_shape() -> None:
    """
    Assert that the second difference matrix has the expected shape.
    """
    matrix = second_difference_matrix(4)

    assert (4, 4) == matrix.shape
