"""Input contract shared by flood scene evaluation and statistical analysis."""

import numpy as np


def validate_labels(prediction, target, reference=None):
    """Return the valid mask; reject invalid values before any dtype conversion.

    Prediction/reference classes are 0 and 1. Targets additionally allow -1
    (ignored). All arrays must be HxW with identical shapes. Validation includes
    ignored pixels because their neighbors participate in boundary construction.
    """
    arrays = [("prediction", prediction, (0, 1)), ("target", target, (-1, 0, 1))]
    if reference is not None:
        arrays.append(("reference", reference, (0, 1)))
    for name, value, allowed in arrays:
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError(f"{name} must be an HxW numpy array")
    for name, value, allowed in arrays:
        if value.shape != target.shape:
            raise ValueError("prediction, target and reference must have identical shapes")
        if value.dtype.kind not in "biuf" or not np.isin(value, allowed).all():
            raise ValueError(f"{name} labels must belong to {allowed}")
    return (target >= 0) & (target < 2)
