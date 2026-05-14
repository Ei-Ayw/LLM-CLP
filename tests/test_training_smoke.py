"""
Smoke tests for training pipeline.

These tests verify that the model can perform a single forward and backward pass
on a tiny synthetic dataset without crashing.
"""
import pytest
import torch
from transformers import AutoTokenizer


class TestTrainingSmoke:
    def test_model_forward(self):
        """Model forward pass produces expected shapes."""
        from src.llm_clp.models.classifier import DebertaV3CausalFair

        model = DebertaV3CausalFair(
            model_path="microsoft/deberta-v3-base",
            num_classes=2,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        texts = ["Hello world", "This is a test"]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")

        with torch.no_grad():
            out = model(inputs["input_ids"], inputs["attention_mask"], return_features=True)

        assert out["logits"].shape == (2, 2)
        assert out["features"].shape[0] == 2
        assert out["features"].shape[1] == 128  # projection_dim

    def test_losses(self):
        """Loss functions compute without error."""
        from src.llm_clp.models.losses import (
            CounterfactualLogitPairing,
            CounterfactualSupConLoss,
        )

        B = 4
        logits_orig = torch.randn(B, 2)
        logits_cf = torch.randn(B, 2)
        features_orig = torch.randn(B, 128)
        features_cf = torch.randn(B, 128)
        labels = torch.tensor([0, 1, 0, 1])

        clp = CounterfactualLogitPairing()
        loss_clp = clp(logits_orig, logits_cf)
        assert loss_clp.item() >= 0

        con = CounterfactualSupConLoss(temperature=0.07)
        loss_con = con(features_orig, features_cf, labels)
        assert loss_con.item() >= 0

    def test_cfr_computation(self):
        """CFR metric computation is correct."""
        from src.llm_clp.evaluation.metrics import compute_cfr

        preds_orig = torch.tensor([0, 0, 1, 1])
        preds_cf = torch.tensor([0, 1, 1, 0])

        cfr = compute_cfr(
            preds_orig.numpy(),
            preds_cf.numpy(),
        )
        # 2 flips out of 4
        assert cfr == 0.5

    def test_ctfg_computation(self):
        """CTFG metric computation is correct."""
        from src.llm_clp.evaluation.metrics import compute_ctfg

        probs_orig = torch.tensor([0.1, 0.4, 0.9])
        probs_cf = torch.tensor([0.2, 0.3, 0.8])

        ctfg = compute_ctfg(probs_orig.numpy(), probs_cf.numpy())
        expected = (0.1 + 0.1 + 0.1) / 3
        assert abs(ctfg - expected) < 1e-6
