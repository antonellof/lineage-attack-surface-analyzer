"""Pure functions of the similarity robustness script."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robustness", ROOT / "scripts/similarity_robustness.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_permutation_p_separates_perfect_order_from_noise():
    x = np.arange(12.0)
    assert mod.permutation_p(x, x, 2000, 0) < 0.001
    assert mod.permutation_p(x, np.tile([1.0, 0.0], 6), 2000, 0) > 0.3


def test_js_divergence_is_zero_for_equal_and_log2_for_disjoint():
    torch = pytest.importorskip("torch")
    p = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1)
    assert float(mod.js_divergence(p, p)) == pytest.approx(0.0, abs=1e-6)
    a = torch.log(torch.tensor([[1.0, 1e-30]]))
    b = torch.log(torch.tensor([[1e-30, 1.0]]))
    assert float(mod.js_divergence(a, b)) == pytest.approx(np.log(2.0), abs=1e-5)
