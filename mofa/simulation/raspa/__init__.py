"""Interfaces to the varied versions of RASPA available"""
from .graspa import gRASPARunner
from .raspa import RASPARunner
from .base import BaseRaspaRunner

runners: dict[str, type[BaseRaspaRunner]] = {
    'graspa': gRASPARunner,
    'raspa2': RASPARunner
}
