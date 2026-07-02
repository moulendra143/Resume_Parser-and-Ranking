"""
Centralised configuration loader.

Reads ``config/config.yaml`` once and exposes a singleton ``AppConfig``
dataclass.  All modules should import ``get_config()`` instead of
hard-coding parameter values directly in source code.

Usage::

    from src.utils.config_loader import get_config

    cfg = get_config()
    weight = cfg.similarity.embedding_weight   # 0.7
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Config schema (mirrors config/config.yaml structure)
# ---------------------------------------------------------------------------

@dataclass
class AppSection:
    name: str = "Resume Screening & Ranking System"
    version: str = "1.0.0"


@dataclass
class PathsSection:
    skills_taxonomy: str = "config/skills_taxonomy.json"
    resumes_dir: str = "data/resumes"
    job_descriptions_dir: str = "data/job_descriptions"
    model_path: str = "model.joblib"


@dataclass
class SimilaritySection:
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_weight: float = 0.7


@dataclass
class RankingSection:
    n_estimators: int = 100
    random_seed: int = 42
    synthetic_samples: int = 1000


@dataclass
class ScoringWeightsSection:
    similarity: float = 0.60
    skills: float = 0.20
    experience: float = 0.10
    education: float = 0.10


@dataclass
class AppConfig:
    """Top-level application configuration container."""
    app: AppSection = field(default_factory=AppSection)
    paths: PathsSection = field(default_factory=PathsSection)
    similarity: SimilaritySection = field(default_factory=SimilaritySection)
    ranking: RankingSection = field(default_factory=RankingSection)
    scoring_weights: ScoringWeightsSection = field(default_factory=ScoringWeightsSection)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "config.yaml"
)


def _load_yaml(path: str) -> dict:
    """Read and parse a YAML file, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Malformed config file at {path!r}: {exc}") from exc


def _build_config(raw: dict) -> AppConfig:
    """Construct an ``AppConfig`` from the raw YAML dict."""
    cfg = AppConfig()

    if "app" in raw:
        cfg.app = AppSection(**{k: v for k, v in raw["app"].items() if hasattr(AppSection(), k)})

    if "paths" in raw:
        cfg.paths = PathsSection(**{k: v for k, v in raw["paths"].items() if hasattr(PathsSection(), k)})

    if "similarity" in raw:
        cfg.similarity = SimilaritySection(
            **{k: v for k, v in raw["similarity"].items() if hasattr(SimilaritySection(), k)}
        )

    if "ranking" in raw:
        cfg.ranking = RankingSection(
            **{k: v for k, v in raw["ranking"].items() if hasattr(RankingSection(), k)}
        )

    if "scoring_weights" in raw:
        cfg.scoring_weights = ScoringWeightsSection(
            **{k: v for k, v in raw["scoring_weights"].items() if hasattr(ScoringWeightsSection(), k)}
        )

    return cfg


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and return the application configuration (singleton via LRU cache).

    Args:
        config_path: Optional path to the YAML config file.  Defaults to
            ``config/config.yaml`` relative to the project root.

    Returns:
        A fully populated :class:`AppConfig` instance.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    raw = _load_yaml(os.path.abspath(path))
    return _build_config(raw)
