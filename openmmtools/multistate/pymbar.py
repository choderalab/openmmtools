from typing import Dict

import numpy as np

try:
    # pymbar >= 4
    from pymbar.other_estimators import exp
    from pymbar.timeseries import (
        detect_equilibration,
        statistical_inefficiency_multiple,
        subsample_correlated_data,
    )
except ImportError:
    # pymbar < 4
    from pymbar import EXP as exp
    from pymbar.timeseries import (
        detectEquilibration as detect_equilibration,
        statisticalInefficiencyMultiple as statistical_inefficiency_multiple,
        subsampleCorrelatedData as subsample_correlated_data,
    )


def _pymbar_bar(
    work_forward: np.ndarray,
    work_backward: np.ndarray,
) -> Dict[str, float]:
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
