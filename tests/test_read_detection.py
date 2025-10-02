import os
from pathlib import Path

import pytest

from src.preprocessing.read_detection import detect_fastq_layout, representative_inputs


@pytest.fixture
def raw_dir(tmp_path):
    # Mirror structure pipeline expects (files directly in directory)
    return tmp_path


def _touch_fastq(path: Path):
    path.write_text("@SEQ\nACGT\n+\nIIII\n")
    return path


def test_detect_single_end_returns_single_list(raw_dir):
    single = _touch_fastq(raw_dir / "sample.fastq")

    layout = detect_fastq_layout(raw_dir)

    assert layout["pairs"] == []
    assert layout["singles"] == [str(single)]


def test_detect_paired_end_r1_r2(raw_dir):
    r1 = _touch_fastq(raw_dir / "sample_R1.fastq")
    r2 = _touch_fastq(raw_dir / "sample_R2.fastq")

    layout = detect_fastq_layout(raw_dir)

    assert layout["pairs"] == [(str(r1), str(r2))]
    assert layout["singles"] == []


def test_detect_paired_end_dot_pattern(raw_dir):
    r1 = _touch_fastq(raw_dir / "sample.1.fastq.gz")
    r2 = _touch_fastq(raw_dir / "sample.2.fastq.gz")

    layout = detect_fastq_layout(raw_dir)

    assert layout["pairs"] == [(str(r1), str(r2))]
    assert layout["singles"] == []


def test_mixed_inputs_unmatched_go_to_singles(raw_dir):
    r1 = _touch_fastq(raw_dir / "paired_R1.fastq")
    r2 = _touch_fastq(raw_dir / "paired_R2.fastq")
    orphan = _touch_fastq(raw_dir / "lonely.fastq")

    layout = detect_fastq_layout(raw_dir)

    assert (str(r1), str(r2)) in layout["pairs"]
    assert str(orphan) in layout["singles"]


def test_representative_inputs_prefers_pairs(raw_dir):
    r1 = _touch_fastq(raw_dir / "paired_R1.fastq")
    r2 = _touch_fastq(raw_dir / "paired_R2.fastq")
    _touch_fastq(raw_dir / "single.fastq")

    layout = detect_fastq_layout(raw_dir)

    rep = representative_inputs(layout)

    assert rep == [str(r1), str(r2)]


def test_representative_inputs_falls_back_to_single(raw_dir):
    single = _touch_fastq(raw_dir / "single.fastq")

    layout = detect_fastq_layout(raw_dir)

    rep = representative_inputs(layout)

    assert rep == [str(single)]
