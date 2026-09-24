import numpy as np
import pytest
from helpers import RATES, SHAPE, smooth_field, split, true_blocks

from skipzfp import codec


@pytest.mark.parametrize("rate", RATES)
def test_metadata_bounds_original_and_reconstructed(rate):
    a = smooth_field()
    buf, cmin, cmax, eps, offs = split(a, rate)
    rec = codec.native().decode(buf, SHAPE, rate, 4)

    assert cmin == a.min() and cmax == a.max()
    assert eps >= np.abs(rec.astype(np.float64) - a).max()

    span = cmax - cmin
    lo = cmin + offs[:, 0] / 255 * span
    hi = cmin + offs[:, 1] / 255 * span
    orig_b, rec_b = true_blocks(a), true_blocks(rec)
    assert np.all(lo <= orig_b.min(1)) and np.all(hi >= orig_b.max(1))
    assert np.all(lo - eps <= rec_b.min(1)) and np.all(hi + eps >= rec_b.max(1))
