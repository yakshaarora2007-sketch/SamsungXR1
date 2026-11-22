# Checker_Harness/config/settings.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class DatasetConfig:
    root_dir: str = "dataset/euroc"
    default_sequence: str = "MH_01_easy"
    segment_duration_s: float = 15.0
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "MH01": "MH_01_easy",
        "MH_01": "MH_01_easy",
        "MH_01_easy": "MH_01_easy",
        "MH02": "MH_02_easy",
        "MH_02": "MH_02_easy",
        "MH_02_easy": "MH_02_easy",
        "MH03": "MH_03_medium",
        "MH04": "MH_04_difficult",
        "MH05": "MH_05_difficult",
        "V101": "V1_01_easy",
        "V1_01_easy": "V1_01_easy",
        "V102": "V1_02_medium",
        "V1_02_medium": "V1_02_medium",
        "V103": "V1_03_difficult",
        "V1_03_difficult": "V1_03_difficult",
    })


@dataclass
class ExperimentConfig:
    output_dir: str = "outputs"
    t1_runs: int = 10
    t2_runs: int = 10
    t3_runs: int = 10
    harness_version: str = "1.0.0"


@dataclass
class HarnessConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    degrader: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "HarnessConfig":
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ds_data = data.get("dataset", {})
        exp_data = data.get("experiment", {})
        deg_data = data.get("degrader", {})

        return cls(
            dataset=DatasetConfig(**ds_data),
            experiment=ExperimentConfig(**exp_data),
            degrader=deg_data,
        )
