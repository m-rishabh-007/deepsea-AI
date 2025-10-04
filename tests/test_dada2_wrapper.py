import subprocess
from pathlib import Path

import pytest

from src.preprocessing.dada2_wrapper import run_dada2


@pytest.fixture
def clean_dir(tmp_path):
    directory = tmp_path / "clean"
    directory.mkdir()
    (directory / "sample_clean.fastq.gz").write_text("@SEQ\nACGT\n+\nIIII\n")
    return directory


@pytest.mark.parametrize("mode", ["single", "paired"], ids=["single-end", "paired-end"])
def test_run_dada2_builds_command_and_returns_paths(monkeypatch, clean_dir, tmp_path, mode):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    captured = {}
    prefix = "test"

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        (output_dir / f"{prefix}_asv_table.csv").write_text("feature,count\nASV1,10\n")
        (output_dir / f"{prefix}_summary.json").write_text("{}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_dada2(
        clean_dir=str(clean_dir),
        output_dir=str(output_dir),
        output_prefix=prefix,
        mode=mode,
        max_ee=3,
        trunc_q=10,
        pool_method="pseudo",
    )

    expected_cmd = [
        "Rscript",
        str(Path("scripts") / "run_dada2.R"),
        str(clean_dir),
        prefix,
        str(output_dir),
        mode,
        "3",
        "10",
        "pseudo",
    ]

    assert captured["cmd"] == expected_cmd
    assert captured["check"] is True

    expected_asv = output_dir / f"{prefix}_asv_table.csv"
    expected_summary = output_dir / f"{prefix}_summary.json"

    assert result == {"asv_table": str(expected_asv), "summary": str(expected_summary)}
    assert expected_asv.exists()
    assert expected_summary.exists()


def test_run_dada2_raises_when_output_missing(monkeypatch, clean_dir, tmp_path):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    def fake_run(cmd, check):
        pass

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        run_dada2(
            clean_dir=str(clean_dir),
            output_dir=str(output_dir),
            output_prefix="missing",
        )


def test_run_dada2_propagates_subprocess_error(monkeypatch, clean_dir, tmp_path):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_dada2(
            clean_dir=str(clean_dir),
            output_dir=str(output_dir),
        )
