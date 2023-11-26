"""Sparse difference matrix definitions."""

from numpy import concatenate, ones, roll, zeros
from scipy.sparse import dia_matrix  # type: ignore

from ..boundary_conditions import BoundaryCondition1D


def second_difference_matrix(
    size: int = 3,
    boundary_condition: BoundaryCondition1D = BoundaryCondition1D.CC,
):
    """Second order difference matrix.
    Generates the second order difference operator in matrix form.

    Keyword arguments:
    size -- the matrix size (default 3)
    boundary
    """
    if boundary_condition == BoundaryCondition1D.FF:
        diag = [
            concatenate((zeros(1), -2 + zeros(size - 1), zeros(1))),
            concatenate((ones(size - 1), zeros(2))),
        ]

        return dia_matrix(([diag[0], roll(diag[1], 2), diag[1]], [0, 1, -1]), shape=(size, size))

    if boundary_condition in (BoundaryCondition1D.CF, BoundaryCondition1D.SF):
        diag = [
            concatenate((-2 + zeros(size - 1), zeros(1))),
            concatenate((ones(size - 1), zeros(1))),
        ]

        return dia_matrix(
            ([diag[0], roll(diag[1], 1), roll(diag[1], -1)], [0, 1, -1]),
            shape=(size, size),
        )

    if boundary_condition in (BoundaryCondition1D.FC, BoundaryCondition1D.FS):
        diag = [
            concatenate((zeros(1), -2 + zeros(size - 1))),
            concatenate((zeros(1), ones(size - 1))),
        ]

        return dia_matrix(
            ([diag[0], roll(diag[1], 1), roll(diag[1], -1)], [0, 1, -1]),
            shape=(size, size),
        )

    diag = [-2 + zeros(size), ones(size)]

    return dia_matrix(([diag[0], diag[1], diag[1]], [0, 1, -1]), shape=(size, size))
