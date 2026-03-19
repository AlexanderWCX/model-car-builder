# ModelCar Builder

Tool for packaging HuggingFace models as OCI container images for deployment on Red Hat OpenShift AI (RHOAI) using KServe ModelCar format.

## Project Structure

- `build.sh` — Main orchestrator script (download, build, archive, save, clean, status)
- `compose.yaml` — Docker/Podman compose for the download stage
- `Containerfile.download` — Download stage image (Python 3.11 + huggingface-hub)
- `Containerfile` — Dynamically generated at build time for the final OCI image
- `download_model.py` — Python script that downloads model weights via huggingface-hub
- `models/` — Transient directory for current model weights (gitignored)
- `models_archive/` — Persistent per-model storage after archiving (gitignored)
- `save/` — Exported OCI images as split tarballs for air-gapped transfer

## Tech Stack

- **Bash** — Main build orchestration
- **Python 3.11** — Model download utility
- **Podman/Docker** — Container runtime
- **UBI9** — Red Hat base images (ubi-micro for final image, python-311 for downloader)
- **huggingface-hub + hf_transfer** — Fast model downloads

## Key Commands

```bash
./build.sh all        # Full pipeline: download → build → archive
./build.sh download   # Download model weights
./build.sh build      # Build OCI image from models/
./build.sh archive    # Move models/ → models_archive/<slug>/
./build.sh save       # Export image as split tarballs
./build.sh clean      # Delete models/
./build.sh status     # Show config and state
```

## Configuration

Top of `build.sh`:
- `MODEL_REPO` — HuggingFace model identifier
- `HF_TOKEN` — Auth token for gated models
- `SPLIT_SIZE` — Split size for save tarballs (default 4G)

## Conventions

- Image naming: slash-to-double-dash (`qwen/model` → `qwen--model`)
- Date tags: YYYYMMDD format
- Each safetensor shard gets its own OCI layer for parallel pulls
- Final images run as user 65534 (nobody), no ENTRYPOINT
- Strict bash: `set -euo pipefail`
- Models path inside container: `/models/`
