"""Public HawkesNest benchmark suite APIs."""
from hawkesnest.suites.base import BaseSuite, GenerationResult
from hawkesnest.suites.entanglement import EntanglementSuite
from hawkesnest.suites.heterogeneity import HeterogeneitySuite
from hawkesnest.suites.recipe import HawkesNestRecipe

__all__ = [
    "BaseSuite",
    "EntanglementSuite",
    "GenerationResult",
    "HawkesNestRecipe",
    "HeterogeneitySuite",
]
