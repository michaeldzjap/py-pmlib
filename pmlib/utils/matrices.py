"""Sparse difference matrix utilities."""

from numpy import array, ones, prod, zeros
from scipy.sparse import dia_matrix, spmatrix  # type: ignore


def second_difference_matrix(n: int = 3) -> spmatrix:
    """
    Generate the second order difference operator in matrix form.

    Parameters
    ----------
    n : int, default 3
        The number of rows and columns of the matrix (default 3).

    Returns
    -------
    spmatrix
        The second difference matrix.
    """
    data = array([-2. + zeros(n), ones(n)])

    return dia_matrix((array([data[0], data[1], data[1]]), array([0, 1, -1])), shape=(n, n))


def fourth_difference_matrix(n: int = 3) -> spmatrix:
    """
    Generate the fourth order difference operator in matrix form.

    Parameters
    ----------
    n : int, default 3
        The number of rows and columns of the matrix (default 3).

    Returns
    -------
    spmatrix
        The fourth difference matrix.
    """
    data = array([6 + zeros(n), -4 + zeros(n), ones(n)])

    return dia_matrix(
        (array([data[0], data[1], data[1], data[2], data[2]]), array([0, 1, -1, 2, -2])),
        shape=(n, n),
    )


def laplacian_matrix(n: tuple[int, int] = (3, 3)) -> spmatrix:
    """
    Generate the discrete laplacian operator in matrix form.

    Parameters
    ----------
    n : tuple, default (3, 3)
        The number of rows and columns of the matrix (default 3).

    Returns
    -------
    spmatrix
        The laplacian matrix.
    """
    m = prod(n)
    m *= m
    data = array([-4 + zeros(m), ones(m)])

    return dia_matrix(
        (array([data[0], data[1], data[1], data[1], data[1]]), array([0, 1, -1, n[1], -n[1]])),
        shape=(m, m),
    )


def biharmonic_matrix(n: tuple[int, int] = (3, 3)) -> spmatrix:
    """
    Generate the discrete biharmonic operator in matrix form.

    Parameters
    ----------
    n : tuple, default (3, 3)
        The number of rows and columns of the matrix (default 3).

    Returns
    -------
    spmatrix
        The biharmonic matrix.
    """
    m = prod(n)
    m *= m
    data = array([20 + zeros(m), -8 + zeros(m), ones(m), 2 + zeros(m)])

    return dia_matrix(
        (
            array([
                data[0], data[1], data[1], data[2], data[2], data[1], data[1],
                data[3], data[3], data[3], data[3], data[2], data[2],
            ]),
            array([
                0, 1, -1, 2, -2, n[1], -n[1], n[1] - 1, n[1] + 1, -n[1] + 1,
                -n[1] - 1, 2 * n[1], 2 * -n[1],
            ]),
        ),
        shape=(m, m),
    )


__all__ = [
    'second_difference_matrix',
    'fourth_difference_matrix',
    'laplacian_matrix',
    'biharmonic_matrix',
]
