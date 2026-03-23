"""Compatibility import for thermal surrogate module layouts.

Supports both:
  - thermal_surrogate.py at repository root
  - agent/thermal_surrogate.py inside an ``agent`` package
"""

from importlib import import_module
from importlib.util import find_spec


def _load_surrogate_module():
    if find_spec('thermal_surrogate') is not None:
        return import_module('thermal_surrogate')

    if find_spec('agent') is not None and \
            find_spec('agent.thermal_surrogate') is not None:
        return import_module('agent.thermal_surrogate')

    raise ModuleNotFoundError(
        'Could not find thermal surrogate module. Expected either '
        '"thermal_surrogate.py" at the project root or '
        '"agent/thermal_surrogate.py" inside an "agent" package.')


EnsembleThermalSurrogate = _load_surrogate_module().EnsembleThermalSurrogate
