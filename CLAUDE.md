# ModelCar Builder

Tool for packaging HuggingFace models and datasets as OCI container images for deployment on Red Hat OpenShift AI (RHOAI) using KServe ModelCar format.

## Project Structure

- `build.sh` — Main orchestrator script (download, build, archive, save, clean, status)
- `Containerfile.download` — Download stage image (Python 3.11 + huggingface-hub + pandas)
- `Containerfile` — Dynamically generated at build time for the final OCI image
- `download_model.py` — Python script that downloads model/dataset files via huggingface-hub
- `reshard_safetensors.py` — Re-splits oversized safetensors into even shards (stdlib-only, crash-safe) so no OCI layer exceeds the registry's per-layer limit
- `convert_parquet.py` — Converts downloaded parquet files to JSONL for datasets
- `models/` — Transient directory for current model weights (gitignored)
- `models_archive/` — Persistent per-model storage after archiving (gitignored)
- `datasets/` — Transient directory for current dataset files (gitignored)
- `datasets_archive/` — Persistent per-dataset storage after archiving (gitignored)
- `save/` — Exported OCI images as split tarballs for air-gapped transfer

## Tech Stack

- **Bash** — Main build orchestration
- **Python 3.11** — Model/dataset download utility
- **Podman/Docker** — Container runtime (podman preferred, docker supported)
- **UBI9** — Red Hat base images (ubi-micro for final image, python-311 for downloader)
- **huggingface-hub + hf_transfer** — Fast model downloads
- **pandas + pyarrow** — Parquet-to-JSONL conversion for datasets

## Key Commands

```bash
./build.sh all        # Full pipeline: download → build → archive
./build.sh download   # Download model weights or dataset
./build.sh reshard    # Re-split oversized safetensors to ≤ MAX_SHARD_SIZE (models only)
./build.sh build      # Build OCI image from staged files (auto-reshards models first)
./build.sh convert    # Convert parquet → JSONL (datasets only)
./build.sh archive    # Move staged files into archive directory
./build.sh restore    # Move files back from archive
./build.sh save       # Export image as split tarballs (--docker forces docker)
./build.sh rehash     # Regenerate checksums for a save directory
./build.sh clean      # Delete staged files (model|dataset|all; default all)
./build.sh status     # Show config and state
```

## Configuration

Top of `build.sh` — set ONE of these, not both:
- `MODEL_REPO` — HuggingFace model identifier
- `DATASET_REPO` — HuggingFace dataset identifier
- `HF_TOKEN` — Auth token for gated repos
- `SPLIT_SIZE` — Split size for save tarballs (default 4G)
- `MAX_SHARD_SIZE` — Max size per safetensors shard / OCI layer (default 4G; empty or `0` disables resharding)

## Conventions

- Image naming: slash-to-double-dash (`qwen/model` → `qwen--model`)
- Date tags: YYYYMMDD format
- Each safetensor shard (or dataset parquet/jsonl shard) gets its own OCI layer for parallel pulls
- Oversized upstream shards are re-split to ≤ `MAX_SHARD_SIZE` before build so no layer exceeds the registry's per-layer cap (e.g. Quay `MAXIMUM_LAYER_SIZE`, default 20G); the index (`model.safetensors.index.json`) is regenerated to match
- A repo may ship multiple independent weight sets, each with its own index (e.g. Mistral's HF `model-*` + `model.safetensors.index.json` AND consolidated `consolidated-*` + `consolidated.safetensors.index.json`, with different tensor naming). The resharder groups safetensors by their index and reshards each group separately, regenerating every index — it never merges the sets
- Final images run as user 65534 (nobody), no ENTRYPOINT
- Strict bash: `set -euo pipefail`
- Paths inside container: `/models/` or `/datasets/`
- Both `.containerignore` and `.dockerignore` exist and are kept in sync (podman reads the former, docker reads the latter)
- All `build` invocations must pass `-f Containerfile` explicitly — docker's classic builder does not auto-detect `Containerfile`
