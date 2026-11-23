"""Typed configuration loaded from config.yaml.

Every section is a dataclass with defaults that match the report, so the code
runs even without a config file. Unknown keys in the YAML raise an error,
which catches typos such as ``sauvola_windw`` before a long run starts.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class PathsCfg:
    data_dir: str = "data"
    interim_dir: str = "data/interim"
    fonts_dir: str = "data/fonts"
    models_dir: str = "models"
    results_dir: str = "results"
    register_csv: str = "data/register.csv"
    rules: str = "rules.yaml"
    db_path: str = "results/lre.db"


@dataclass
class FontsCfg:
    train: list[str] = field(default_factory=lambda: [
        "Kalam-Light.ttf", "Kalam-Regular.ttf", "Kalam-Bold.ttf", "Amita-Regular.ttf"])
    test: list[str] = field(default_factory=lambda: ["Tillana-Regular.ttf", "Tillana-Medium.ttf"])
    printed: str = "Hind-Regular.ttf"


@dataclass
class SynthCfg:
    dpi: int = 300
    page_width_in: float = 14.0
    page_height_in: float = 10.0
    image_format: str = "png"
    khewats_per_village: list[int] = field(default_factory=lambda: [80, 150])
    owners_per_holding: list[int] = field(default_factory=lambda: [1, 4])
    khasras_per_holding: list[int] = field(default_factory=lambda: [1, 5])
    contradiction_rate: float = 0.12
    benign_rate: float = 0.20
    page_error_rate: float = 0.02
    renumber_rate: float = 0.05
    split_fractions: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    fonts: FontsCfg = field(default_factory=FontsCfg)
    font_px: list[int] = field(default_factory=lambda: [34, 46])
    slant_deg: list[float] = field(default_factory=lambda: [-4.0, 14.0])
    ascii_digit_prob: float = 0.6
    max_skew_deg: float = 5.0
    jpeg_quality: list[int] = field(default_factory=lambda: [40, 90])
    blur_prob: float = 0.5
    stain_prob: float = 0.5
    bleed_prob: float = 0.4
    fade_prob: float = 0.5


@dataclass
class PreCfg:
    background_kernel: int = 51
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    nlmeans_h: int = 10
    deskew_max_deg: float = 10.0
    sauvola_window: int = 31
    sauvola_k: float = 0.2


@dataclass
class LayoutCfg:
    n_columns: int = 12
    h_kernel_frac: int = 30
    v_kernel_frac: int = 40
    cell_padding: int = 4
    line_height: int = 64
    width_buckets: list[int] = field(default_factory=lambda: [128, 256, 384, 512])


@dataclass
class DecodeCfg:
    beam_width: int = 10
    lexicon_weight: float = 0.6


@dataclass
class LinkCfg:
    min_khasra_jaccard: float = 0.5
    strong_khasra_jaccard: float = 0.8
    min_owner_similarity: float = 0.6


@dataclass
class RouteCfg:
    alpha: float = 0.98
    audit_sample: float = 0.02
    contradiction_theta: Any = "from_validation"


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    synth: SynthCfg = field(default_factory=SynthCfg)
    preprocess: PreCfg = field(default_factory=PreCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    decoding: DecodeCfg = field(default_factory=DecodeCfg)
    linking: LinkCfg = field(default_factory=LinkCfg)
    routing: RouteCfg = field(default_factory=RouteCfg)

    def hash(self) -> str:
        """Short, stable hash written into every output for traceability (report E.3)."""
        blob = yaml.safe_dump(dataclasses.asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


class ConfigError(ValueError):
    pass


def _merge(obj: Any, data: dict, where: str) -> Any:
    hints = get_type_hints(type(obj))
    known = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown config key '{where}.{key}'")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"'{where}.{key}' must be a mapping")
            _merge(current, value, f"{where}.{key}")
        else:
            if hints.get(key) is float and isinstance(value, int):
                value = float(value)
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = "config.yaml") -> Config:
    """Load config.yaml on top of the defaults. A missing file gives the defaults."""
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at top level")
    return _merge(cfg, data, "config")
