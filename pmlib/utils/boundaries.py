"""Boundary condition utilities."""

from enum import Enum, auto

from scipy.sparse import spmatrix


class BoundaryCondition(Enum):
    """All possible boundary conditions."""
    CLAMPED = auto()
    SIMPLY_SUPPORTED = auto()
    FREE = auto()


def apply_boundary_condition(matrix: spmatrix, conditions: tuple[BoundaryCondition]) -> spmatrix:
    """
    Apply boundary conditions to a matrix.

    Parameters
    ----------
    matrix : spmatrix
        The matrix to which to apply the boundary condition.
    condition : tuple
        The boundary condition to apply to the matrix.

    Returns
    -------
    spmatrix
        The original matrix with the boundary condition applied.
    """
    pass


__all__ = [
    'BoundaryCondition',
]
