from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite


SIMULATOR_CLASS = "hawkesnest.simulator.hawkes.HawkesSimulator"


def test_entanglement_suite_generate_smoke():
    result = EntanglementSuite().generate(level="L2", n_events=50, seed=123)

    assert len(result.events) == 50
    assert result.simulator_class_name == SIMULATOR_CLASS
    assert result.metadata["simulator_class"] == SIMULATOR_CLASS


def test_heterogeneity_suite_generate_smoke():
    result = HeterogeneitySuite().generate(level="H3", n_events=50, seed=123)

    assert len(result.events) == 50
    assert result.simulator_class_name == SIMULATOR_CLASS
    assert result.metadata["simulator_class"] == SIMULATOR_CLASS


def test_export_smoke(tmp_path: Path):
    out_dir = tmp_path / "export"
    result = EntanglementSuite().generate(level="L2", n_events=50, seed=123, out_dir=out_dir)

    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "metadata.json").exists()
    assert result.export_paths["jsonl"] == str(out_dir / "events.jsonl")

    with (out_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert len(row["times"]) == 50
    assert len(row["locations"]) == 50

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["simulator_class_name"] == SIMULATOR_CLASS


def test_cli_generate_entanglement_smoke(tmp_path: Path):
    out_dir = tmp_path / "entanglement"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hawkesnest.cli",
            "generate",
            "entanglement",
            "--level",
            "L2",
            "--n-events",
            "50",
            "--seed",
            "123",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "metadata.json").exists()


def test_cli_generate_heterogeneity_smoke(tmp_path: Path):
    out_dir = tmp_path / "heterogeneity"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hawkesnest.cli",
            "generate",
            "heterogeneity",
            "--level",
            "H3",
            "--n-events",
            "50",
            "--seed",
            "123",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "events.csv").exists()
    assert (out_dir / "metadata.json").exists()
