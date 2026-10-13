"""
AI aircraft-design automation module for aeronautics research.

Turns natural-language mission requirements into an optimized, constraint-verified
conventional fixed-wing aircraft design, using AeroSandbox + NeuralFoil as the
analysis backend.

This package is *not* part of the distributed `aerosandbox` wheel; it lives at the
repository top level and consumes the flat `aerosandbox` (`asb`) API only.
"""

__version__ = "0.1.0"

from design_agent.mission_spec import MissionSpec

__all__ = ["MissionSpec", "__version__"]
