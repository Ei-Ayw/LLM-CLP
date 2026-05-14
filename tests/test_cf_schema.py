"""
Tests for counterfactual schema validation.
"""
import pytest
from src.llm_clp.counterfactual.schema import (
    CounterfactualPair,
    validate_cf_pair_schema,
)


class TestCounterfactualPair:
    def test_to_dict_roundtrip(self):
        pair = CounterfactualPair(
            original_text="Muslims are destroying this country",
            cf_text="Christians are destroying this country",
            source_group="muslim",
            target_group="christian",
            method="llm",
            post_id="sample_0",
        )
        d = pair.to_dict()
        restored = CounterfactualPair.from_dict(d)

        assert restored.original_text == pair.original_text
        assert restored.cf_text == pair.cf_text
        assert restored.source_group == pair.source_group
        assert restored.method == pair.method

    def test_validate_schema_valid(self):
        data = {
            "original_text": "Black people are lazy",
            "cf_text": "White people are lazy",
            "source_group": "black",
            "target_group": "white",
        }
        errors = validate_cf_pair_schema(data)
        assert errors == []

    def test_validate_schema_missing_original(self):
        data = {"cf_text": "Christians are destroying this country"}
        errors = validate_cf_pair_schema(data)
        assert any("original_text" in e for e in errors)

    def test_validate_schema_missing_cf(self):
        data = {"original_text": "Some text"}
        errors = validate_cf_pair_schema(data)
        assert any("cf_text" in e for e in errors)

    def test_validate_schema_empty_string(self):
        data = {"original_text": "Some text", "cf_text": "   "}
        errors = validate_cf_pair_schema(data)
        assert any("non-empty" in e for e in errors)

    def test_to_json_from_json(self):
        pair = CounterfactualPair(
            original_text="Test text",
            cf_text="Test text modified",
            method="swap",
        )
        s = pair.to_json()
        restored = CounterfactualPair.from_json(s)
        assert restored.original_text == pair.original_text
        assert restored.cf_text == pair.cf_text
