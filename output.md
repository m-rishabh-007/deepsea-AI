# DeepSea-AI Stage 1 Outputs

This document captures every artifact produced by the Stage 1 preprocessing pipeline (fastp → DADA2 → k-mer vectorization) and explains how to interpret those files across common run scenarios (successful run, low-yield data, and failures).

---

## Pipeline Directories

| Directory | Typical Path | Contents | Notes |
|-----------|---------------|----------|-------|
| Raw inputs | `data/raw/fastq_dataset/` *(API jobs: `data/jobs/<job_id>/raw/`)* | Unmodified FASTQ files (uploaded or bundled samples). | Immutable reference copy; never overwritten by pipeline stages. |
| Interim workspace | `data/interim/` *(API jobs: `data/jobs/<job_id>/interim/`)* | fastp cleaned reads, temp QC assets. | Safe to delete between runs once results are archived. |
| Processed results | `data/processed/` *(API jobs: `data/jobs/<job_id>/processed/`)* | DADA2 outputs, k-mer vectors, run metadata. | Long-term home for machine-learning inputs. |
| Logs | `logs/` | `pipeline.log` plus rotating copies. | Central log stream for both Streamlit "direct" and API jobs. |

In Streamlit direct mode, per-run temporary folders are created under `/tmp` (e.g., `/tmp/interim`, `/tmp/processed`). The metadata file written in the processed directory points back to these transient locations.

---

## Step-by-Step Outputs

### 1. fastp Quality Control

| Artifact | Location | Description | Interpretation |
|----------|----------|-------------|----------------|
| `*_clean.fastq.gz` | Interim directory | Quality-filtered reads ready for DADA2. Naming mirrors the original FASTQ, suffixed with `_clean`. | Higher retention ⇒ better downstream sensitivity. A steep drop indicates aggressive filters or poor input quality. |
| `fastp_report.json` | Interim directory | Machine-readable QC summary. Includes per-cycle quality stats, filtering counts, adapter trimming data. | Use to drive dashboards or automated alerts. `filtering_result` signals why reads were discarded. |
| `fastp_report.html` | Interim directory & project root (copied by Streamlit for download). | Interactive QC dashboard with adapter/quality plots. | First stop when investigating quality drops; download via Streamlit "Quality Report" tab. |

**Scenarios**
- *Ideal quality*: High `before_filtering` vs `after_filtering` parity, Q30 > 85%.
- *Adapter/quality issues*: Large `low_quality_reads` or `too_short_reads`; cross-check trimming settings.
- *Pipeline disabled fastp*: `fastp` section in metadata becomes `null`; downstream steps will consume raw input.

### 2. DADA2 Amplicon Sequence Variants

| Artifact | Location | Description | Interpretation |
|----------|----------|-------------|----------------|
| `dada2_asv_table.csv` | Processed directory | Sequences (`sequence`) with abundance (`count`). | Primary biological output. Zero rows ⇒ no confident ASVs (check quality or truncation lengths). |
| `dada2_summary.json` | Processed directory | Totals: `total_asvs`, `total_reads`, mean/median counts, processing mode. | Use in UI metrics and alerts (e.g., `total_reads` drop). |
| `*_filt.fastq.gz` / `*_F_filt.fastq.gz`, `*_R_filt.fastq.gz` | Processed directory (created during filtering) | DADA2-filtered reads used for error learning. | Retained for audits; size disparity between F/R can hint at dropout. |

**Scenarios**
- *Healthy diversity*: `total_asvs` proportional to sample complexity; inspect count distribution.
- *Single ASV retained*: Often due to strict filtering; consider relaxing `maxEE` or `truncLen` in `pipeline.yaml`.
- *Empty ASV table*: Indicates complete read loss or chimera removal; revisit fastp output and raw data.

### 3. K-mer Vectorization

| Artifact | Location | Description | Interpretation |
|----------|----------|-------------|----------------|
| `kmer_vectors.csv` (default filename configurable) | Processed directory | Wide CSV: one row per ASV sequence, columns for each k-mer feature plus `sequence` and `count`. | Feed directly into ML pipelines; monitor sparsity for model readiness. |
| `kmer_metadata.json` *(if enabled in config)* | Processed directory | Describes k-mer settings (`k`, normalization) and feature ordering. | Critical when exporting vectors outside this system; ensures reproducible feature mapping. |

**Scenarios**
- *Normalized vectors*: Default pipeline writes relative frequencies (sum ≈ 1). |
- *Raw counts*: When `normalize: false`, expect integer counts; update downstream expectations. |
- *Disabled k-mer step*: Metadata notes `"kmer": null`; only ASV outputs available.

### 4. Run Metadata & Logs

| Artifact | Location | Description | Uses |
|----------|----------|-------------|------|
| `stage1_metadata.json` | Processed directory | Master manifest bundling paths, timestamp, mode, and nested step outputs. | Streamlit UI and API responses read from here; use as authoritative record of a run. |
| `pipeline.log` | `logs/` | Rolling log file with timestamps, run IDs, and exception traces. | Inspect when pipeline fails or stalls. |
| Job database entries | PostgreSQL (`pipeline_jobs` table) | For API runs: status, directories, metadata JSON, error string. | Enables REST queries (`GET /jobs/<id>`). |

---

## Streamlit & API Surfaces

### Streamlit Direct Mode
- **UI Metrics**: Derived live from `meta` (e.g., ASV count, k-mer feature count).
- **Download buttons**:
  - "Download ASV Table" → Serializes `dada2_asv_table.csv`.
  - "Download K-mer Vectors" → Serializes `kmer_vectors.csv`.
  - "Download HTML Report" → Streams `fastp_report.html`.
- **Session State**: Stores last `meta`, `interim_dir`, `processed_dir` to rehydrate "Previous Results" section.

### API Mode Outputs
- `POST /jobs`: Creates job directories and DB record.
- `POST /jobs/<id>/files`: Returns uploaded file list.
- `POST /jobs/<id>/run`: Starts background pipeline; job status flips to `RUNNING`.
- `GET /jobs/<id>`: Returns DB record including `meta` once complete.
- `GET /jobs/<id>/vectors`: Streams k-mer CSV for download.
- `GET /jobs/<id>/metadata`: Full metadata payload (mirrors `stage1_metadata.json`).

---

## Scenario Playbook

| Situation | Symptoms in Outputs | Recommended Checks |
|-----------|---------------------|--------------------|
| **Successful run** | `stage1_metadata.json` present; `fastp` reports high retention; `dada2_asv_table.csv` populated; `kmer_vectors.csv` available. | Archive processed directory, push vectors to downstream ML. |
| **Low-quality reads** | `fastp_report.json` → high `low_quality_reads` / `too_short_reads`; cleaned FASTQs much smaller than raw. | Review raw data; adjust `qualified_quality_phred`, `length_required`, or sequencing pipeline. |
| **No ASVs detected** | `dada2_summary.json` shows `total_asvs = 0`; `dada2_asv_table.csv` empty. | Inspect fastp output, consider relaxing DADA2 filtering (`maxEE`, `truncLen`). |
| **Pipeline failure** | `pipeline.log` contains traceback; API job status `FAILED` with `error` field; `stage1_metadata.json` missing. | Use log message to locate failing step (fastp binary, R environment, file naming). |
| **K-mer step skipped** | Metadata shows `"kmer": null`; `kmer_vectors.csv` absent. | Ensure DADA2 succeeded and `kmer.enabled` is `true`. |
| **Paired-end mismatch** | Pipeline halts before fastp; error mentions missing R2. | Check raw directory naming conventions (`*_R1/_R2` or `.1/.2`). |

---

## How to Use This Document

1. **During Development**: Confirm new code paths still populate the artifacts listed here; update this file with any additional outputs.
2. **During Operations**: Use the scenario table to triage pipeline issues quickly. Share the relevant artifact (`fastp_report.html`, `dada2_summary.json`, etc.) with upstream sequencing teams as evidence.
3. **For Handoffs**: Pair this document with `README.md` to give downstream ML engineers clarity on what data to expect from Stage 1.

---

## Next Steps & Extensions

- Add automated validation scripts that parse `stage1_metadata.json` and surface warnings when counts fall below thresholds.
- Consider versioning output schemas (e.g., include a `schema_version` field) so downstream components can detect breaking changes.
- For long-term storage, move processed directories into object storage (S3, GCS) and retain only `stage1_metadata.json` pointers locally.
