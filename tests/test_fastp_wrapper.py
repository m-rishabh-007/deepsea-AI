import subprocess
from pathlib import Path

import pytest

from src.preprocessing.fastp_wrapper import run_fastp


def _touch(path: Path):
    path.write_text("@SEQ\nACGT\n+\nIIII\n")
    return path


@pytest.mark.parametrize(
    "input_names, kwargs, expect_adapter_flag",
    [
        (
            ["sample.fastq"],
            {"threads": 8, "qualified_quality_phred": 20, "length_required": 70},
            False,
        ),
        (
            ["sample_R1.fastq", "sample_R2.fastq"],
            {"threads": 6, "json_report": "report.json", "html_report": "report.html"},
            True,
        ),
        (
            ["sample_R1.fastq", "sample_R2.fastq"],
            {"detect_adapter_for_pe": False},
            False,
        ),
    ],
    ids=["single-end", "paired-adapter", "paired-no-adapter"],
)
def test_run_fastp_commands(monkeypatch, tmp_path, input_names, kwargs, expect_adapter_flag):
    inputs = [_touch(tmp_path / name) for name in input_names]
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        captured["check"] = check
        Path(cmd[cmd.index("-o") + 1]).write_text("")
        if "-O" in cmd:
            Path(cmd[cmd.index("-O") + 1]).write_text("")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_fastp([str(path) for path in inputs], str(tmp_path), **kwargs)

    threads = kwargs.get("threads", 4)
    qqp = kwargs.get("qualified_quality_phred", 15)
    length_req = kwargs.get("length_required", 50)
    json_report = kwargs.get("json_report", "fastp_report.json")
    html_report = kwargs.get("html_report", "fastp_report.html")

    expected_cmd = [
        "fastp",
        "-i", str(inputs[0]),
    ]
    if len(inputs) == 2:
        expected_cmd.extend(["-I", str(inputs[1])])

    out_paths = []
    first_clean = tmp_path / f"{inputs[0].stem}_clean.fastq.gz"
    expected_cmd.extend(["-o", str(first_clean)])
    out_paths.append(str(first_clean))
    if len(inputs) == 2:
        second_clean = tmp_path / f"{inputs[1].stem}_clean.fastq.gz"
        expected_cmd.extend(["-O", str(second_clean)])
        out_paths.append(str(second_clean))

    expected_cmd.extend(
        [
            "-w",
            str(threads),
            "-j",
            str(tmp_path / json_report),
            "-h",
            str(tmp_path / html_report),
            "-q",
            str(qqp),
            "-l",
            str(length_req),
        ]
    )

    if expect_adapter_flag:
        expected_cmd.append("--detect_adapter_for_pe")

    assert captured["cmd"] == expected_cmd
    assert captured["check"] is True

    if expect_adapter_flag:
        assert "--detect_adapter_for_pe" in captured["cmd"]
    else:
        assert "--detect_adapter_for_pe" not in captured["cmd"]

    assert set(result["clean_reads"]) == set(out_paths)
    assert result["json_report"] == str(tmp_path / json_report)
    assert result["html_report"] == str(tmp_path / html_report)


def test_run_fastp_raises_on_subprocess_failure(monkeypatch, tmp_path):
    input_fastq = _touch(tmp_path / "sample.fastq")

    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_fastp([str(input_fastq)], str(tmp_path))


def test_run_fastp_rejects_invalid_input_count(tmp_path):
    with pytest.raises(ValueError):
        run_fastp([], str(tmp_path))

    with pytest.raises(ValueError):
        run_fastp(["a", "b", "c"], str(tmp_path))
