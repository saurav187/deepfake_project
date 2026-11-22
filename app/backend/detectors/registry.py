import importlib
from importlib.util import find_spec

from ..config import Settings
from .base import Detector
from .mock import MockDetector


class BackendUnavailable(RuntimeError):
    pass


def load_backend(settings: Settings) -> Detector:
    if settings.backend == 'mock':
        return MockDetector()
    if settings.backend == 'legacy':
        from .legacy import LegacyDetector

        return LegacyDetector(settings.device)
    if settings.backend == 'detector':
        try:
            spec = find_spec('detector.inference')
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            raise BackendUnavailable('DETECTOR_BACKEND=detector requires detector/inference.py.')
        module = importlib.import_module('detector.inference')
        if not callable(getattr(module, 'load_detector', None)):
            raise BackendUnavailable('detector.inference must expose load_detector().')
        return module.load_detector()
    raise BackendUnavailable('DETECTOR_BACKEND must be mock, legacy, or detector.')
