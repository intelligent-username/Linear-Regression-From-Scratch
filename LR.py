"""Backward-compatible module.

The original single-file implementation has been moved into the package
`naivelinear`. Import from here for legacy scripts:

    from LR import linear_regression

Prefer new import path:

    from naivelinear import linear_regression
"""
from naivelinear import linear_regression  # re-export

__all__ = ["linear_regression"]

