import json
from pathlib import Path
from datetime import datetime

import pytest

from src import pipeline


@pytest.fixture
def sample_config(tmp_path, monkeypatch):
    cfg = {
        "paths": {
            "logs_dir": str(tmp_path / "logs")
        },
        "fastp": {
            "enabled": True,
            "threads": 4,
            "qualified_quality_phred": 15,
            "length_required": 50,
            "detect_adapter_for_pe": True,
            "json_report": "fastp_report.json",
            "html_report": "fastp_report.html"
        },
        "dada2": {
            "enabled": True,
            "output_prefix": "dada2",
            "max_ee": 2,
            "trunc_q": 2,
            "pool_method": "pseudo"
        },
        "kmer": {
            "enabled": True,
            "k": 6,
            "normalize": True,
            "output_vectors": "kmer_vectors.csv"
        },
        "logging": {
            "level": "INFO"
        }
    }
    config_path = tmp_path / "pipeline.yaml"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    def fake_load_config(_path):
        return cfg

    monkeypatch.setattr(pipeline, "load_config", fake_load_config)
    return cfg


@pytest.fixture
def raw_dir(tmp_path):
    path = tmp_path / "raw"
    path.mkdir()
    (path / "sample_R1.fastq").write_text("@SEQ\nACGT\n+\nIIII\n")
    (path / "sample_R2.fastq").write_text("@SEQ\nTGCA\n+\nIIII\n")
    return path


@pytest.fixture
def interim_dir(tmp_path):
    path = tmp_path / "interim"
    path.mkdir()
    return path


@pytest.fixture
def processed_dir(tmp_path):
    path = tmp_path / "processed"
    path.mkdir()
    return path


@pytest.fixture
def mock_fastp(monkeypatch, tmp_path):
    outputs = {
        "clean_reads": [str(tmp_path / "sample_R1_clean.fastq.gz"), str(tmp_path / "sample_R2_clean.fastq.gz")],
        "json_report": str(tmp_path / "fastp_report.json"),
        "html_report": str(tmp_path / "fastp_report.html")
    }

    def fake_fastp(input_fastq, output_dir, **kwargs):
        for clean in outputs["clean_reads"]:
            Path(clean).write_text("")
        Path(outputs["json_report"]).write_text("{}")
        Path(outputs["html_report"]).write_text("<html></html>")
        return outputs

    monkeypatch.setattr(pipeline, "run_fastp", fake_fastp)
    return outputs


@pytest.fixture
def mock_dada2(monkeypatch, tmp_path):
    outputs = {
        "asv_table": str(tmp_path / "dada2_asv_table.csv"),
        "summary": str(tmp_path / "dada2_summary.json")
    }

    def fake_dada2(clean_dir, output_dir, **kwargs):
        Path(outputs["asv_table"]).write_text("sequence,count\nACGT,10\n")
        Path(outputs["summary"]).write_text("{}")
        return outputs

    monkeypatch.setattr(pipeline, "run_dada2", fake_dada2)
    return outputs


@pytest.fixture
def mock_vectorizer(monkeypatch, tmp_path):
    import pandas as pd

    def fake_vectorize(asv_table, k, normalize):
        return pd.DataFrame([
            {"sequence": "ACGT", "count": 10, "AAAAAA": 0.1}
        ])

    def fake_save_vectors(df, output_csv):
        Path(output_csv).write_text(df.to_csv(index=False))

    monkeypatch.setattr(pipeline, "vectorize_asv_table", fake_vectorize)
    monkeypatch.setattr(pipeline, "save_vectors", fake_save_vectors)


def test_pipeline_happy_path(sample_config, raw_dir, interim_dir, processed_dir, mock_fastp, mock_dada2, mock_vectorizer, monkeypatch, tmp_path):
    meta = pipeline.run_pipeline(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        config_path=str(tmp_path / "irrelevant.yaml"),
    )

    assert meta["fastp"] == mock_fastp
    assert meta["dada2"] | {"mode": "paired"} == meta["dada2"]
    assert meta["kmer"]["vectors_csv"].endswith("kmer_vectors.csv")
    assert (Path(processed_dir) / "kmer_vectors.csv").exists()
    assert (Path(processed_dir) / "stage1_metadata.json").exists()


def test_pipeline_handles_missing_fastq(sample_config, interim_dir, processed_dir, monkeypatch, tmp_path):
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()

    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(
            raw_dir=empty_raw,
            interim_dir=interim_dir,
            processed_dir=processed_dir,
        )


def test_pipeline_skips_steps(sample_config, raw_dir, interim_dir, processed_dir, monkeypatch, tmp_path):
    cfg = sample_config
    cfg["fastp"]["enabled"] = False
    cfg["dada2"]["enabled"] = False
    cfg["kmer"]["enabled"] = False

    def fake_load_config(_path):
        return cfg

    monkeypatch.setattr(pipeline, "load_config", fake_load_config)

    meta = pipeline.run_pipeline(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
    )

    assert meta["fastp"] is None
    assert meta["dada2"] is None
    assert meta["kmer"] is None
    assert (Path(processed_dir) / "stage1_metadata.json").exists()
