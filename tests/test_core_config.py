"""Tests for the normalized core configuration boundary."""

import pytest

from src.core import CoreConfig, classify_config_fields, config_digest


def test_from_mapping_matches_legacy_normalization():
    config = {
        "alpha": 0.7,
        "preprocessing": {"roi": False, "resize": 0.5, "debug": True},
        "visualization": {"alpha": 0.7, "labels": "cat, dog"},
        "postsam2processing": {"maxsize": 500},
    }
    core = CoreConfig.from_mapping(config)
    assert core.alpha == 0.7
    assert core.roi_val is None  # legacy false -> None normalization
    assert core.resize_val == 0.5
    assert core.prep_debug is True
    assert core.keep_labels == ("cat", "dog")
    assert core.post_maxsize == 500
    assert core.max_w == 999_999_999


def test_from_mapping_requires_alpha_like_legacy():
    with pytest.raises(KeyError):
        CoreConfig.from_mapping({"preprocessing": {}})


def test_classification_partitions_top_level_fields():
    classification = classify_config_fields(
        {
            "preprocessing": {},
            "clip": {},
            "images": {"type": "jpg"},
            "export_yolo_det": {"labels": "cat"},
            "something_new": 1,
        }
    )
    assert classification.algorithm_fields == ("clip", "preprocessing")
    assert classification.batch_only_fields == ("export_yolo_det", "images")
    assert classification.unrecognized_fields == ("something_new",)
    assert classification.as_dict()["unrecognized"] == ["something_new"]


def test_config_digest_is_deterministic_and_sensitive():
    base = {
        "alpha": 0.6,
        "preprocessing": {"roi": None, "resize": None},
        "visualization": {"labels": ["cat"]},
    }
    first = CoreConfig.from_mapping(base)
    second = CoreConfig.from_mapping(dict(base))
    assert config_digest(first) == config_digest(second)
    assert config_digest(first) != config_digest(
        first.__class__.from_mapping({**base, "alpha": 0.9})
    )
