class ModelLoadError(RuntimeError):
    """Artifacts are missing, tampered with, or built with incompatible library versions."""


class DatasetError(ValueError):
    """An uploaded dataset cannot be processed (bad file, missing columns, ...)."""
