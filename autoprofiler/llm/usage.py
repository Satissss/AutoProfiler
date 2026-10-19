"""Backward-compat shim: Usage is now Meter in autoprofiler.metering."""
from autoprofiler.metering import Meter as Usage

__all__ = ["Usage"]
