"""
Tests for fairness metrics.
"""
import numpy as np
import pytest
from src.llm_clp.evaluation.metrics import (
    compute_cfr,
    compute_ctfg,
    compute_fped_fned,
)


class TestComputeCFR:
    def test_cfr_no_flips(self):
        preds_orig = np.array([0, 0, 1, 1])
        preds_cf = np.array([0, 0, 1, 1])
        assert compute_cfr(preds_orig, preds_cf) == 0.0

    def test_cfr_all_flips(self):
        preds_orig = np.array([0, 1])
        preds_cf = np.array([1, 0])
        assert compute_cfr(preds_orig, preds_cf) == 1.0

    def test_cfr_half_flips(self):
        preds_orig = np.array([0, 0, 1, 1])
        preds_cf = np.array([0, 1, 1, 0])
        assert compute_cfr(preds_orig, preds_cf) == 0.5

    def test_cfr_empty(self):
        assert compute_cfr(np.array([]), np.array([])) == 0.0


class TestComputeCTFG:
    def test_ctfg_no_difference(self):
        probs_orig = np.array([0.1, 0.5, 0.9])
        probs_cf = np.array([0.1, 0.5, 0.9])
        assert compute_ctfg(probs_orig, probs_cf) == 0.0

    def test_ctfg_uniform_difference(self):
        probs_orig = np.array([0.2, 0.4])
        probs_cf = np.array([0.3, 0.5])
        # mean(|0.2-0.3|, |0.4-0.5|) = mean(0.1, 0.1) = 0.1
        assert compute_ctfg(probs_orig, probs_cf) == pytest.approx(0.1)

    def test_ctfg_empty(self):
        assert compute_ctfg(np.array([]), np.array([])) == 0.0


class TestComputeFPEDFNED:
    def test_fped_fned_equal_groups(self):
        # All predictions correct, equal across groups
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        groups = np.array(["A", "A", "B", "B"])
        fped, fned, details = compute_fped_fned(y_true, y_pred, groups)

        assert fped == 0.0
        assert fned == 0.0
        assert "A" in details
        assert "B" in details

    def test_fped_fned_imbalanced(self):
        # Group A: 0 FP, Group B: 1 FP out of 2 negatives each
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])  # 1 FP for group B
        groups = np.array(["A", "A", "B", "B"])
        fped, fned, details = compute_fped_fned(y_true, y_pred, groups)

        # Overall FPR = 1/4 = 0.25, Group A FPR = 0, Group B FPR = 1/2 = 0.5
        # FPED = |0-0.25| + |0.5-0.25| = 0.5
        assert details["A"]["fpr"] == 0.0
        assert details["B"]["fpr"] == 0.5
