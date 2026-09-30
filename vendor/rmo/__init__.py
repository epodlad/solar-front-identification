"""RMO synthetic solver prototype conforming to RMO-R6A-SPEC-1.0.0."""

from .models import State, Wave, Solution, SolveResult
from .solver import RiemannSolver

__all__ = ["State", "Wave", "Solution", "SolveResult", "RiemannSolver"]

__version__ = "0.1.0"
