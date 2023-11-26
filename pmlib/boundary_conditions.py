"""Boundary condition definitions."""

from enum import Enum, auto


class BoundaryCondition1D(Enum):
    """All possible combinations of 1D boundary conditions."""
    CC = auto()  # Left clamped, right clamped
    CS = auto()  # Left clamped, right simply supported
    SC = auto()  # Left simply supported, right clamped
    SS = auto()  # Left simply supported, right simply supported
    CF = auto()  # Left clamped, right free
    FC = auto()  # Left free, right clamped
    FF = auto()  # Left free, right free
    SF = auto()  # Left simply supported, right free
    FS = auto()  # Left free, right simply supported
