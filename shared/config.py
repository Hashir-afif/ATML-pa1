"""YAML config loading: a run config is ``base.yaml`` deep-merged with a method file."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(config_dir: Path, name: str, base: str = "base.yaml") -> dict:
    config_dir = Path(config_dir)
    cfg = _merge(yaml.safe_load((config_dir / base).read_text()), yaml.safe_load((config_dir / f"{name}.yaml").read_text()))
    cfg["config_files"] = [f"{config_dir.name}/{base}", f"{config_dir.name}/{name}.yaml"]
    return cfg
