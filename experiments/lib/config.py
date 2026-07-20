"""
Experiment configuration system.

Loads typed config from YAML files. Validates required fields at load time.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    artifacts_dir: str
    results_dir: str


@dataclass
class EmbeddingConfig:
    model: str
    normalize: bool = True


@dataclass
class NullConfig:
    z_score_fraction: float = 0.8


@dataclass
class HCConfig:
    gamma: float = 0.1
    pool_size: int = 1000


@dataclass
class MethodConfig:
    name: str
    type: str
    k: int | None = None


@dataclass
class EvalConfig:
    qa_model: str
    judge_model: str
    qa_prompt: str
    judge_prompt: str
    title_truncate: int = 100
    delay: float = 0.3


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    null: NullConfig
    hc: HCConfig
    methods: list[MethodConfig]
    eval: EvalConfig


def load_config(path: str) -> ExperimentConfig:
    """Load YAML config, validate, return typed ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    dataset = DatasetConfig(**raw["dataset"])
    embedding = EmbeddingConfig(**raw.get("embedding", {}))
    null_cfg = NullConfig(**raw.get("null", {}))
    hc_cfg = HCConfig(**raw.get("hc", {}))

    methods = [MethodConfig(**m) for m in raw["methods"]]
    for m in methods:
        if m.type not in ("baseline", "hc"):
            raise ValueError(f"Unknown method type '{m.type}' in method '{m.name}'")
        if m.type == "baseline" and m.k is None:
            raise ValueError(f"Baseline method '{m.name}' requires 'k' parameter")

    eval_cfg = EvalConfig(**raw["eval"])

    return ExperimentConfig(
        dataset=dataset,
        embedding=embedding,
        null=null_cfg,
        hc=hc_cfg,
        methods=methods,
        eval=eval_cfg,
    )
