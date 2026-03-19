# ModelCar Builder

Build OCI container images from any HuggingFace model for deployment on Red Hat OpenShift AI (RHOAI) using KServe's ModelCar capabilities.

The entire process runs inside containers -- no local Python installation required. Works with both Podman and Docker (prefers Podman, falls back to Docker).

## Prerequisites

- Podman (or Docker)
- Sufficient disk space for the model weights (2x the model size to account for build layers)
- Internet access to `registry.access.redhat.com` and `huggingface.co`
- (Optional) `pv` for progress bars: `apt install pv` / `dnf install pv`

## Quick start

Edit `build.sh` and set `MODEL_REPO` to the HuggingFace model you want to package. If the model is gated, set `HF_TOKEN` to your HuggingFace token.

Then run:

```bash
./build.sh all
```

This will:

1. Build a temporary downloader image.
2. Download the model weights into `models/`.
3. Build the ModelCar OCI image with per-shard layers.
4. Move the weights into `models_archive/<model-slug>/` so `models/` is clean for the next build.

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
  download  Download model weights into models/
  build     Build the ModelCar OCI image from models/
              Optional flag: ./build.sh build --single-layer
  archive   Move models/ into models_archive/<model-slug>/
  restore   Move weights from models_archive/ back into models/
              Optionally pass a path: ./build.sh restore models_archive/<dir>
  save      Save the image as split tar files for air-gapped transfer
              Optionally pass an image tag: ./build.sh save <image:tag>
  rehash    Regenerate checksums for a save directory
              Optionally pass a path: ./build.sh rehash save/<dir>
  clean     Delete all downloaded weights from models/
  status    Show current configuration and state
```

### Individual steps

Each step can be run independently:

```bash
./build.sh download                 # download weights
./build.sh build                    # build OCI image (per-shard layers)
./build.sh build --single-layer     # build OCI image (single layer)
./build.sh archive                  # move weights to archive
./build.sh restore                  # move weights back from archive
./build.sh clean                    # delete contents of models/
./build.sh status                   # show current state
```

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

### Checksums

Both BLAKE2 (`b2sum`) and SHA-256 (`sha256sum`) checksums are generated in parallel during save. The load script auto-detects which is available, preferring BLAKE2 for speed.

To regenerate checksums for existing save directories:

```bash
./build.sh rehash                          # rehash all save directories
./build.sh rehash save/qwen--qwen3-vl...   # rehash a specific one
```

### Cross-platform compatibility

The `save` and `load` scripts auto-detect Podman or Docker. Images saved with Docker can be loaded with Podman and vice versa.

Note: When Podman is detected, `save` uses `--format=oci-archive` (supports zstd-compressed layers). When Docker is detected, it uses the default Docker archive format. Both formats are loadable by both runtimes.

### Using podman/docker directly

```bash
# Internet-connected side
podman save --format=oci-archive -o model.tar <image-name>:<tag>

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
├── build.sh                   # Main script: download, build, archive, save, rehash, etc.
├── Containerfile              # Generated at build time (one layer per safetensor shard)
├── Containerfile.download     # Downloader image (Python + huggingface-hub + hf_transfer)
├── download_model.py          # Download script (configurable via env vars)
├── models/                    # Downloaded weights (gitignored except .gitkeep)
│   └── .gitkeep
├── models_archive/            # Archived weights per model (gitignored)
├── save/                      # Split image tarballs for air-gapped transfer (gitignored)
│   └── <model-slug>/
│       ├── checksums.b2       # BLAKE2 checksums (if b2sum available)
│       ├── checksums.sha256   # SHA-256 checksums
│       ├── load.sh            # Self-contained verify/assemble/load script
│       └── model.tar.part*    # Split image parts
├── .containerignore           # Excludes archive/save/cache dirs from build context
├── .gitignore
└── README.md
```

## References

- [RHOAI: Storing a model in an OCI image](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.22/html/serving_models/serving-large-models_serving-large-models#storing-a-model-in-oci-image_serving-large-models)
- [Build and deploy a ModelCar container in OpenShift AI](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)