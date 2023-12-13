"""Resonator definitions."""

from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt
from typing import NamedTuple

from scipy.sparse import dok_matrix, identity, spmatrix

from .utils.boundaries import BoundaryCondition
from .utils.matrices import fourth_difference_matrix, second_difference_matrix


class Endpoint(Enum):
    """All possible endpoints."""
    LEFT = 'left'
    RIGHT = 'right'


class BoundaryConditions(NamedTuple):
    """Boundary conditions named tuple."""
    left: BoundaryCondition
    right: BoundaryCondition


class Resonator(ABC):
    """
    Abstract base class from which all concrete resonator classes inherit.
    """
    MIN_SAMPLE_RATE = 441
    MAX_SAMPLE_RATE = 705600

    _sample_rate = 44100
    _nyquist = _sample_rate / 2
    _time_step = 1 / _sample_rate

    def __init__(self, gamma: float, kappa: float, loss: tuple[float, float]):
        self.gamma = gamma
        self.kappa = kappa
        self.loss = loss
        self._b = None
        self._c = None

    @property
    def gamma(self) -> float:
        """Get the spatially scaled wave speed."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value < 0 or value > self._nyquist:
            raise ValueError(f'gamma out of the range 0 - {self._nyquist}')

        self._gamma: float = value

    @property
    def kappa(self) -> float:
        """Get the spatially scaled stiffness coefficient."""
        return self._kappa

    @kappa.setter
    def kappa(self, value: float):
        if value < 0:
            raise ValueError('kappa must be equal or greater than 0')

        self._kappa = value

    @property
    def loss(self) -> tuple[float, float]:
        """Get the frequency independent and dependent damping constants."""
        return self._loss

    @loss.setter
    def loss(self, value: tuple[float, float]):
        for i, k in enumerate(value):
            if k < 0:
                raise ValueError(f'loss[{i}] must be equal or greater than 0')

        self._loss = value

    @property
    @abstractmethod
    def b(self) -> spmatrix:
        """Get the update matrix acting on u^n."""
        return self._b

    @property
    @abstractmethod
    def c(self) -> spmatrix:
        """Get the update matrix acting on u^(n - 1)."""
        return self._c

    @classmethod
    def sample_rate(cls, value: int) -> None:
        """
        Set the sample rate.

        Attributes
        ----------
        value : int
            The new sample rate.

        Raises
        ------
        ValueError
            If `value` is smaller than the minimum allowed or greater than the maximum allowed sample rate.
        """
        if value < cls.MIN_SAMPLE_RATE or value > cls.MAX_SAMPLE_RATE:
            raise ValueError(
                f'sample rate out of the range {cls.MIN_SAMPLE_RATE} - {cls.MAX_SAMPLE_RATE}'
            )

        cls._sample_rate = value
        cls._nyquist = value / 2
        cls._time_step = 1 / value

    @abstractmethod
    def build(self) -> None:
        """Build the resonator model."""

    @abstractmethod
    def _compute_grid_step(self) -> tuple[float, int]:
        pass


class LinearResonator1D(Resonator):
    """
    A concrete implementation of a one dimenstional linear resonator.

    Parameters
    ----------
    gamma : float, default 200
        The spatially scaled wave speed.
    kappa : float, default 1
        The spatially scaled stiffness coefficient.
    loss : tuple, default (0, 0)
        The frequency independent and dependent damping constants.
    boundary_conditions : NamedTuple
        The left and right boundary conditions.
    """
    def __init__(
        self,
        gamma: float = 200,
        kappa: float = 1,
        loss: tuple[float, float] = (0, 0),
        boundary_conditions: BoundaryConditions = BoundaryConditions(
            left=BoundaryCondition.SIMPLY_SUPPORTED,
            right=BoundaryCondition.SIMPLY_SUPPORTED,
        ),
    ):
        super().__init__(gamma, kappa, loss)

        self._boundary_conditions = boundary_conditions

    def build(self) -> None:
        h, n = self._compute_grid_step()

        self._b, self._c = self.__build_update_matrices(h, n)

    def _compute_grid_step(self) -> tuple[float, int]:
        k = self._time_step
        gamma = self._gamma
        kappa = self._kappa
        _, b2 = self._loss

        a = (gamma * k) ** 2 + 4. * b2 * k
        h = sqrt(0.5 * (a + sqrt(a ** 2 + 16. * (kappa * k) ** 2)))
        n = int(1 / h)

        return 1 / n, n

    def __build_update_matrices(self, h: float, n: int) -> tuple[spmatrix, spmatrix]:
        k = self._time_step
        gamma = self._gamma
        kappa = self._kappa
        b1, b2 = self._loss

        lambda2 = (gamma * k / h) ** 2
        h2 = h ** 2
        mu2 = (kappa * k / h2) ** 2
        zeta = 2 * b2 * k / h2
        den = 1 + b1 * k

        alpha = 2 * b2 * (h / kappa) ** 2 / k
        beta = 1 + (gamma * h / kappa) ** 2 + alpha

        i = identity(n + 1)
        dxx = second_difference_matrix(n + 1)
        dxxxx = fourth_difference_matrix(n + 1)

        i, dxx, dxxxx = self.__apply_boundary_conditions((i, dxx, dxxxx), beta)

        b = self.__create_b_matrix((i, dxx, dxxxx), (lambda2, zeta, mu2)) / den
        c = self.__create_c_matrix((i, dxx), (b1, k, zeta, alpha, mu2), n) / -den

        return b, c

    def __create_b_matrix(
        self, matrices: tuple[spmatrix, spmatrix, spmatrix], coefficients: tuple[float, float, float],
    ) -> spmatrix:
        i, dxx, dxxxx = matrices
        lambda2, zeta, mu2 = coefficients

        return 2 * i + (lambda2 + zeta) * dxx - mu2 * dxxxx

    def __create_c_matrix(
        self, matrices: tuple[spmatrix, spmatrix], coefficients: tuple[float, ...], n: int,
    ) -> spmatrix:
        conditions = self._boundary_conditions
        i, dxx = matrices
        b1, k, zeta, alpha, mu2 = coefficients

        c = (1. - b1 * k) * i + zeta * dxx

        if any(c == BoundaryCondition.FREE for c in conditions):
            c = self.__add_free_matrix(c, n, (alpha, mu2))

        return c

    def __add_free_matrix(self, c: spmatrix, n: int, coefficients: tuple[float, float]):
        conditions = self._boundary_conditions
        alpha, mu2 = coefficients

        m = dok_matrix((n + 1, n + 1))

        if conditions.left == BoundaryCondition.FREE:
            m[0, 0] = -alpha
            m[0, 1] = alpha

        if conditions.right == BoundaryCondition.FREE:
            m[-1, -1] = -alpha
            m[-1, -2] = alpha

        return c + mu2 * m.todia()

    def __apply_boundary_conditions(
        self, matrices: tuple[spmatrix, spmatrix, spmatrix], beta: float,
    ) -> tuple[spmatrix, ...]:
        conditions = self._boundary_conditions

        i, dxx, dxxxx = [m.todok() for m in matrices]

        for endpoint in Endpoint:
            condition = getattr(conditions, endpoint.value)

            if condition == BoundaryCondition.CLAMPED:
                self.__apply_clamped_condition((i, dxx, dxxxx), endpoint)

            if condition == BoundaryCondition.SIMPLY_SUPPORTED:
                self.__apply_simply_supported_condition((i, dxx, dxxxx), endpoint)

            if condition == BoundaryCondition.FREE:
                self.__apply_free_condition((i, dxx, dxxxx), endpoint, beta)

        return tuple(m.todia() for m in (i, dxx, dxxxx))

    def __apply_clamped_condition(self, matrices: tuple[spmatrix, spmatrix, spmatrix], endpoint: Endpoint) -> None:
        i, dxx, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            i[0, 0] = 0

            dxx[0, 0:2] = 0
            dxx[1, 0] = 0

            dxxxx[0, 0:3] = 0
            dxxxx[1, 0] = 0
            dxxxx[2, 0] = 0

        if endpoint == Endpoint.RIGHT:
            i[-1, -1] = 0

            dxx[-1, -2:] = 0
            dxx[-2, -1] = 0

            dxxxx[-1, -3:] = 0
            dxxxx[-2, -1] = 0
            dxxxx[-3, -1] = 0

    def __apply_simply_supported_condition(
        self, matrices: tuple[spmatrix, spmatrix, spmatrix], endpoint: Endpoint,
    ) -> None:
        self.__apply_clamped_condition(matrices, endpoint)

        _, _, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            dxxxx[1, 1] = 5

        if endpoint == Endpoint.RIGHT:
            dxxxx[-2, -2] = 5

    def __apply_free_condition(
        self, matrices: tuple[spmatrix, spmatrix, spmatrix], endpoint: Endpoint, beta: float,
    ) -> None:
        _, dxx, dxxxx = matrices

        if endpoint == Endpoint.LEFT:
            dxx[0, 0:2] = 0

            dxxxx[0, 0] = beta
            dxxxx[0, 1] = -1 - beta
            dxxxx[1, 0] = -2
            dxxxx[1, 1] = 5

        if endpoint == Endpoint.RIGHT:
            dxx[-1, -2:] = 0

            dxxxx[-1, -1] = beta
            dxxxx[-1, -2] = -1 - beta
            dxxxx[-2, -1] = -2
            dxxxx[-2, -2] = 5


__all__ = ['LinearResonator1D']
