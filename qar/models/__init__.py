"""Volatility forecasting models: a common interface, baselines, and the research agenda."""

from qar.models.base import FitResult, Forecaster
from qar.models.baselines import BASELINES, EWMAVariance, GARCH11, RandomWalkVariance
from qar.models.research import GJRGarch, HARRV, MarkovSwitchingVariance, NeuralVolatility

__all__ = [
    "FitResult", "Forecaster",
    "BASELINES", "EWMAVariance", "GARCH11", "RandomWalkVariance",
    "GJRGarch", "HARRV", "MarkovSwitchingVariance", "NeuralVolatility",
]
