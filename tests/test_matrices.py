"""Test sparse difference matrix utilities."""

from numpy import array
from numpy.testing import assert_array_equal

from pmlib.utils.matrices import second_difference_matrix


def test_second_difference_matrix() -> None:
    """
    Assert that the second difference matrix contains the expected
    coefficients at all relevant indices.
    """
    matrix = second_difference_matrix()

    assert_array_equal(array([-2, -2, -2]), matrix.A.diagonal(0))
