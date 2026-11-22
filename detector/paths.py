"""Filesystem locations. Large data and run outputs stay out of Git."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DETECTOR = REPO / "detector"
DATASETS = DETECTOR / "datasets"
RUNS = DETECTOR / "runs"
LEGACY = REPO / "deepfake_model"

ASVSPOOF5 = DATASETS / "asvspoof5"
ASVSPOOF5_PROTOCOLS = ASVSPOOF5 / "protocols"
IN_THE_WILD_ZIP = DATASETS / "in_the_wild" / "release_in_the_wild.zip"
INDIEFAKE = LEGACY / "datasets" / "indiefake"
VA_DATASET = LEGACY / "datasets" / "va_dataset"

MANIFESTS = RUNS / "manifests"
AUDIT = RUNS / "audit"
