from typing import Sequence

import numpy as np


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(np.array(values), p))
