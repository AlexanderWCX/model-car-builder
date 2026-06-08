# ModelCar Builder

Build OCI container images from any HuggingFace model or dataset for deployment on Red Hat OpenShift AI (RHOAI) using KServe's ModelCar capabilities.

The entire process runs inside containers -- no local Python installation required. Works with both Podman and Docker (prefers Podman, falls back to Docker).

## Prerequisites

- Podman (or Docker)
- Sufficient disk space for the model weights (2x the model size to account for build layers)
- Internet access to `registry.access.redhat.com` and `huggingface.co`
- (Optional) `pv` for progress bars: `apt install pv` / `dnf install pv`

## Quick start

Edit `build.sh` and set `MODEL_REPO` (or `DATASET_REPO` for datasets) at the top. Only set one at a time. If the repo is gated, set `HF_TOKEN`.

Then run:

```bash
./build.sh all
```

This will:

1. Build a temporary downloader image.
2. Download the model weights (or dataset) into the staging directory.
3. Build the OCI image with per-shard layers.
4. Build a tokenizer-only image (models only).
5. Move the files into the archive directory so staging is clean for the next build.

Then push to your registry:

```bash
podman tag <image-tag> <your-registry>/<image-name>:<tag>
podman push <your-registry>/<image-name>:<tag>
```

Override the image tag if needed:

```bash
IMAGE_TAG=my-model:v1 ./build.sh all
```

## Commands

Run `./build.sh` with no arguments to see all available commands:

```
Usage: ./build.sh <command>

Commands:
  all       Run the full pipeline (download -> build -> archive)
  download  Download model weights or dataset into staging directory
  build     Build the OCI image from staged files
              Optional flag: ./build.sh build --single-layer
              (models are auto-resharded first unless MAX_SHARD_SIZE is empty)
  reshard   Re-split oversized safetensors so each shard <= MAX_SHARD_SIZE
              (regenerates model.safetensors.index.json; models only)
  archive   Move staged files into archive directory
  restore   Move files from archive back into staging directory
              Optionally pass a path: ./build.sh restore <archive-dir>
  clean     Delete all files from staging directory
  convert   Convert parquet files to JSONL (datasets only)
  save      Save an image as split tar files for air-gapped transfer
              Optionally pass an image tag: ./build.sh save <image:tag>
              Force docker as the runtime:  ./build.sh save --docker <image:tag>
  rehash    Regenerate checksums for a save directory
              Optionally pass a path: ./build.sh rehash save/<dir>
  status    Show current configuration and state
```

All commands automatically detect whether you're working with a model or dataset based on which config variable is set (`MODEL_REPO` or `DATASET_REPO`). Only one can be set at a time.

### Individual steps

Each step can be run independently:

```bash
./build.sh download                 # download model/dataset
./build.sh reshard                  # re-split oversized safetensors (≤ MAX_SHARD_SIZE)
./build.sh build                    # build OCI image (per-shard layers; auto-reshards first)
./build.sh build --single-layer     # build OCI image (single layer)
./build.sh archive                  # move staged files to archive
./build.sh restore                  # move files back from archive
./build.sh clean                    # delete contents of staging directory
./build.sh status                   # show current state
```

### Build modes

The default build creates one OCI layer per safetensor shard. This enables parallel registry pulls and per-layer resumability, and reduces temp disk space needed during build.

The `--single-layer` flag copies all model files in a single `COPY` instruction. The image tag is suffixed with `-single` to distinguish it:

```
./build.sh build                  -> qwen/qwen3-reranker-4b:20260319
./build.sh build --single-layer   -> qwen/qwen3-reranker-4b:20260319-single
```

### Resharding oversized weights

Each safetensors file becomes its own OCI layer. Some upstream repos pack most of
their weights into a single huge file (e.g. one ~50 GB shard), which produces one
giant layer that **exceeds container-registry per-layer limits** — Quay's
`MAXIMUM_LAYER_SIZE` defaults to `20G` — and also defeats parallel pulls.

Before building (model mode, non `--single-layer`), `build.sh` automatically
re-splits any oversized safetensors into even shards no larger than
`MAX_SHARD_SIZE` (default `4G`) and regenerates `model.safetensors.index.json` to
match. Run it on its own with `./build.sh reshard`.

Details:

- **Lossless & dtype-agnostic.** Weights are repacked at the raw-byte level (no
  torch/numpy), so bf16/fp8/any dtype is copied verbatim.
- **Multiple weight sets.** Some repos ship the model more than once with
  separate indices — e.g. Mistral models carry both the HF format
  (`model-*.safetensors` + `model.safetensors.index.json`) and the consolidated
  format (`consolidated-*.safetensors` + `consolidated.safetensors.index.json`),
  with *different* tensor naming. Each index plus the shards it references is
  treated as an independent group: groups are resharded separately (never merged)
  and **each** index is regenerated against its own new shards. All formats are
  kept, so the resulting image works with either loader. (This roughly doubles
  the on-disk size, so a dual-format repo needs ~its full size in free scratch
  space during resharding — checked up front.)
- **Idempotent.** If every shard is already ≤ `MAX_SHARD_SIZE`, it does nothing.
  A single tensor larger than the target can't be split — it gets its own shard
  and a warning is printed.
- **Crash-safe.** New shards are written into a `.reshard-staging/` subdirectory
  while the originals are left untouched; the swap happens only after a
  `COMMITTED` marker is in place. An interrupted run either discards the
  incomplete staging (originals intact) or finishes the commit on the next run.
  This matters because `restore` *moves* weights out of the archive, so `models/`
  can be the only copy.
- **Disk.** The staged copy briefly coexists with the originals, so resharding
  needs roughly the model's size in additional free space (checked up front).
- Set `MAX_SHARD_SIZE` empty or `0` to disable resharding entirely.

### Tokenizer image

Every build automatically creates a tokenizer-only image tagged `<image-tag>-tokenizer` containing just `tokenizer.json`, `tokenizer_config.json`, and `config.json`. This is useful for attaching to a vLLM instance for benchmarking without loading the full model weights.

If no tokenizer files are found, the tokenizer image is skipped with a warning.

## Datasets

To package a HuggingFace dataset as an OCI image, set `DATASET_REPO` (and clear `MODEL_REPO`) at the top of `build.sh`:

```bash
# In build.sh:
MODEL_REPO=""
DATASET_REPO="lmarena-ai/VisionArena-Chat"
```

Then use the same commands:

```bash
./build.sh all                     # download -> build -> archive
./build.sh save <dataset-tag>      # split + checksum for air-gapped transfer
```

Parquet files get individual layers (same pattern as safetensor shards). The dataset image tag is derived from `DATASET_REPO` with a date stamp.

## Air-gapped transfer

### Using `./build.sh save` (recommended)

The `save` command saves the image as split tar files with checksums, sized for transfer over slow or limited connections:

```bash
./build.sh save
```

This creates a `save/<model-slug>/` directory containing split parts, checksums, and a `load.sh` script. Transfer all files to the air-gapped host.

On the air-gapped side:

```bash
./load.sh           # show help
./load.sh verify    # check checksums only
./load.sh assemble  # check checksums and reassemble into model.tar
./load.sh load      # check checksums and load into podman/docker
```

Configure the split size (default `4G`) by editing `SPLIT_SIZE` at the top of `build.sh`.

To save an arbitrary image (not just the configured model):

```bash
./build.sh save nginx:latest
./build.sh save registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.5
```

The tokenizer image is automatically saved alongside the model image when present.

### Checksums

Both BLAKE2 (`b2sum`) and SHA-256 (`sha256sum`) checksums are generated in parallel during save. The load script auto-detects which is available, preferring BLAKE2 for speed.

To regenerate checksums for existing save directories:

```bash
./build.sh rehash                          # rehash all save directories
./build.sh rehash save/qwen--qwen3-vl...   # rehash a specific one
```

### Cross-platform compatibility

The `save` and `load` scripts auto-detect Podman or Docker. Images saved with Docker can be loaded with Podman and vice versa.

When Podman is detected, `save` uses `--format=docker-archive` (the only format `docker load` accepts). When Docker is detected, it uses the default Docker archive format. Both produce archives loadable by both runtimes.

If you need to force docker as the save runtime (e.g. podman is installed but you want a docker-native archive), pass `--docker`:

```bash
./build.sh save --docker
./build.sh save --docker <image:tag>
```

### Using podman/docker directly

```bash
# Internet-connected side
podman save --format=docker-archive -o model.tar <image-name>:<tag>

# Sneakernet the tar file across

# Air-gapped side
podman load -i model.tar
podman tag <image-name>:<tag> <internal-registry>/<image-name>:<tag>
podman push <internal-registry>/<image-name>:<tag>
```

### Using skopeo

```bash
# Internet-connected side
skopeo copy containers-storage:<image-name>:<tag> oci-archive:model.tar

# Air-gapped side
skopeo copy oci-archive:model.tar docker://<internal-registry>/<image-name>:<tag>
```

## Deploying on RHOAI

### Via the Dashboard

1. Create or select a data science project.
2. Deploy a model using the **single-model serving platform**.
3. Select your vLLM ServingRuntime.
4. For model location, select **URI - v1** and enter:
   ```
   oci://<your-registry>/<image-name>:<tag>
   ```
5. For a private registry, use **OCI compliant registry - v1** and provide credentials.

### Via the CLI

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    openshift.io/display-name: <model-display-name>
    serving.knative.openshift.io/enablePassthrough: "true"
    sidecar.istio.io/inject: "true"
    sidecar.istio.io/rewriteAppHTTPProbers: "true"
  name: <model-name>
  labels:
    opendatahub.io/dashboard: "true"
spec:
  predictor:
    annotations:
      serving.knative.dev/progress-deadline: "30m"
    maxReplicas: 1
    minReplicas: 1
    model:
      modelFormat:
        name: vLLM
      name: ""
      resources:
        limits:
          nvidia.com/gpu: "<gpu-count>"
        requests:
          nvidia.com/gpu: "<gpu-count>"
      runtime: <your-vllm-runtime-name>
      storage:
        key: <your-oci-connection-name>
        path: ""
    tolerations:
      - effect: NoSchedule
        key: nvidia.com/gpu
        operator: Exists
```

> **Note:** The `progress-deadline` of 30 minutes is important -- the first pull
> of large model images can exceed the default 10-minute KNative timeout.

## File structure

```
.
|-- build.sh                   # Main script: download, build, archive, save, rehash, etc.
|-- Containerfile              # Generated at build time (one layer per shard)
|-- Containerfile.download     # Downloader image (Python + huggingface-hub + hf_transfer + pandas)
|-- download_model.py          # Download script (supports models and datasets)
|-- reshard_safetensors.py     # Re-splits oversized safetensors into even shards (crash-safe)
|-- convert_parquet.py         # Parquet-to-JSONL converter (datasets only)
|-- models/                    # Model weight staging area (gitignored except .gitkeep)
|   +-- .gitkeep
|-- datasets/                  # Dataset staging area (gitignored except .gitkeep)
|   +-- .gitkeep
|-- models_archive/            # Archived model weights (gitignored)
|-- datasets_archive/          # Archived datasets (gitignored)
|-- save/                      # Split image tarballs for air-gapped transfer (gitignored)
|   +-- <model-slug>/
|       |-- checksums.b2       # BLAKE2 checksums (if b2sum available)
|       |-- checksums.sha256   # SHA-256 checksums
|       |-- load.sh            # Self-contained verify/assemble/load script
|       +-- model.tar.part*    # Split image parts
|-- .containerignore           # Build context excludes (podman)
|-- .dockerignore              # Build context excludes (docker) -- kept in sync with .containerignore
|-- .gitignore
+-- README.md
```

## References

- [RHOAI: Storing a model in an OCI image](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.22/html/serving_models/serving-large-models_serving-large-models#storing-a-model-in-oci-image_serving-large-models)
- [Build and deploy a ModelCar container in OpenShift AI](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)