"""Tests for the config system."""

import pytest
import yaml
from experiments.lib.config import load_config, ExperimentConfig


@pytest.fixture
def sample_config_path(tmp_path):
    config = {
        "dataset": {
            "name": "test_dataset",
            "data_dir": "datasets/test",
            "artifacts_dir": "artifacts/test",
            "results_dir": "results/test",
        },
        "embedding": {"model": "text-embedding-3-small"},
        "null": {"z_score_fraction": 0.8},
        "hc": {"gamma": 0.1, "pool_size": 1000},
        "methods": [
            {"name": "baseline_k20", "type": "baseline", "k": 20},
            {"name": "hc_default", "type": "hc"},
        ],
        "eval": {
            "qa_model": "gpt-4o-mini",
            "judge_model": "gpt-4o",
            "qa_prompt": "qa_product_titles",
            "judge_prompt": "judge_default",
        },
    }
    path = tmp_path / "test.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_load_valid_config(sample_config_path):
    config = load_config(str(sample_config_path))
    assert isinstance(config, ExperimentConfig)
    assert config.dataset.name == "test_dataset"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.hc.gamma == 0.1
    assert len(config.methods) == 2
    assert config.methods[0].type == "baseline"
    assert config.methods[0].k == 20
    assert config.methods[1].type == "hc"
    assert config.methods[1].k is None


def test_config_defaults(sample_config_path):
    config = load_config(str(sample_config_path))
    assert config.embedding.normalize is True
    assert config.eval.title_truncate == 100
    assert config.eval.delay == 0.3


def test_config_missing_required_field(tmp_path):
    config = {"dataset": {"name": "test"}}
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    with pytest.raises((KeyError, TypeError)):
        load_config(str(path))
