"""
particle_twin.library — per-engine reference library + similarity matching.

Consolidates the Day 10/11/12 work (experiments/day10/similarity_library_prototype.py's
per-engine library + MMD matching + one-shot/periodic streaming, and the
PMMH joint-calibration follow-up) into reusable classes:

  ReferenceLibrary          — build/calibrate/match a per-engine library.
  EngineRecord              — one library entry (growth_rate, sigma_v, HI, source).
  SimilarityStreamingFilter — one-shot/periodic health-index prediction
                              driven by a ReferenceLibrary.
"""

from particle_twin.library.reference_library import ReferenceLibrary, EngineRecord
from particle_twin.library.streaming_filter import SimilarityStreamingFilter

__all__ = ["ReferenceLibrary", "EngineRecord", "SimilarityStreamingFilter"]