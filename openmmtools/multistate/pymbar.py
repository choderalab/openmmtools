from typing import Dict, Tuple

import numpy as np

try:
    # pymbar >= 4
    from pymbar.timeseries import (
        detect_equilibration,
        statistical_inefficiency_multiple,
        subsample_correlated_data,
        statistical_inefficiency
    )
    from pymbar import MBAR, __version__
    from pymbar.utils import ParameterError
except ImportError:
    # pymbar < 4
    from pymbar.timeseries import (
        detectEquilibration as detect_equilibration,
        statisticalInefficiencyMultiple as statistical_inefficiency_multiple,
        subsampleCorrelatedData as subsample_correlated_data,
        statisticalInefficiency as statistical_inefficiency
    )
    from pymbar import MBAR
    from pymbar.utils import ParameterError
    from pymbar.version import short_version as __version__


def _pymbar_bar(
    work_forward: np.ndarray,
    work_backward: np.ndarray,
) -> dict[str, float]:
    """
    https://github.com/shirtsgroup/physical_validation/blob/v1.0.5/physical_validation/util/ensemble.py#L37
    """
    import pymbar

    try:
        # pymbar >= 4
        return pymbar.other_estimators.bar(work_forward, work_backward)
    except AttributeError:
        # pymbar < 4
        return pymbar.BAR(work_forward, work_backward, return_dict=True)

def _pymbar_exp(
        w_F: np.ndarray,
) -> tuple[float, float]:
    try:
        # pymbar < 4
        from pymbar import EXP
        fe_estimate = EXP(w_F)
        return fe_estimate[0], fe_estimate[1]
    except ImportError:
        # pymbar >= 4
        from pymbar.other_estimators import exp
        fe_estimate = exp(w_F)
        return fe_estimate["Delta_f"], fe_estimate["dDelta_f"]
