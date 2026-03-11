# ModelCar Builder

Build OCI container images from any HuggingFace model for deployment on Red Hat OpenShift AI (RHOAI) using KServe's ModelCar capabilities.

The entire process runs inside containers — no local Python installation required.

## Prerequisites

- Podman with the Docker-compatible API socket enabled (`systemctl --user enable --now podman.socket`), or Docker with `docker compose`
  - Alternatively, install [`podman-compose`](https://github.com/containers/podman-compose) (`apt install podman-compose` / `pip install podman-compose`) and replace `podman compose` with `podman-compose` in the commands below
- Sufficient disk space for the model weights (2x the model size to account for build layers)
- Internet access to `registry.access.redhat.com` and `huggingface.co`

## Quick start

Edit the configuration at the top of `build.sh`:

```bash
MODEL_REPO="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
HF_TOKEN=""        # Set for gated models
SPLIT_SIZE="4G"    # Split size for air-gapped exports
```

Then run the full pipeline:

```bash
./build.sh all
```

This will:

1. Download the model weights into `models/`.
2. Build the ModelCar OCI image.
3. Move the weights into `models_archive/<model-slug>/` so `models/` is clean for the next build.

Override the image tag if needed:

```bash
IMAGE_TAG=my-model:v1 ./build.sh all
```

Then push to your registry:

```bash
podman tag <image-tag> quay.io/<your-registry>/<image-name>:<tag>
podman push quay.io/<your-registry>/<image-name>:<tag>
```

## Commands

```
./build.sh <command>

all       Run the full pipeline (download -> build -> archive)
download  Download model weights into models/
build     Build the ModelCar OCI image from models/
archive   Move models/ into models_archive/<model-slug>/
save      Save the image as split tar files for air-gapped transfer
          Optionally pass an image tag: ./build.sh save <image:tag>
rehash    Regenerate checksums for save directories
          Optionally pass a path: ./build.sh rehash save/<dir>
          With no argument, rehashes all directories under save/
clean     Delete all downloaded weights from models/
status    Show current configuration and state
```

## Air-gapped transfer

### Using build.sh (recommended)

Export the image as split tarballs with checksums:

```bash
./build.sh save
```

This creates a directory under `save/` containing split tar files, checksums (SHA-256 and BLAKE2 if `b2sum` is available), and a `load.sh` helper script.

Transfer the entire directory to the air-gapped host, then:

```bash
# Verify checksums only
./load.sh verify

# Verify and load into podman
./load.sh load

# Verify and reassemble into a single tar file
./load.sh assemble
```

To regenerate checksums after a transfer or modification:

```bash
./build.sh rehash save/<dir>

# Or rehash all save directories at once
./build.sh rehash
```

### Using podman directly

```bash
# Internet-connected side
podman save -o model.tar quay.io/<your-registry>/<image-name>:<tag>

# Sneakernet the tar file across

# Air-gapped side
podman load -i model.tar
podman tag quay.io/<your-registry>/<image-name>:<tag> \
  <internal-registry>/<image-name>:<tag>
podman push <internal-registry>/<image-name>:<tag>
```

### Using skopeo

Preserves manifest digests, no local container storage needed:

```bash
# Internet-connected side
skopeo copy \
  containers-storage:quay.io/<your-registry>/<image-name>:<tag> \
  oci-archive:model.tar

# Air-gapped side
skopeo copy \
  oci-archive:model.tar \
  docker://<internal-registry>/<image-name>:<tag>
```

## Manual workflow

If you prefer to run each step individually instead of using `build.sh`:

### 1. Download weights

Edit `compose.yaml` and set `MODEL_REPO` to the HuggingFace model you want to package. If the model is gated, set `HF_TOKEN` to your HuggingFace token.

```bash
podman compose up
```

```bash
ls -lh models/
```

You should see `.safetensors` weight shards, `config.json`, `tokenizer.json`, and related files.

### 2. Build the ModelCar image

```bash
podman build --format=oci \
  -t quay.io/<your-registry>/<image-name>:<tag> .
```

### 3. Archive the model weights

```bash
mkdir -p models_archive/<model-slug>
mv models/* models_archive/<model-slug>/
touch models/.gitkeep
```

To restore archived weights for a rebuild later:

```bash
cp -r models_archive/<model-slug>/* models/
```

### 4. Push to your registry

```bash
podman login quay.io
podman push quay.io/<your-registry>/<image-name>:<tag>
```

## Using Docker instead of Podman

Replace `podman compose` with `docker compose` and `podman build` with `docker build`. Drop the `--format=oci` flag as Docker builds OCI format by default:

```bash
docker compose up
docker build -t <your-registry>/<image-name>:<tag> .
docker push <your-registry>/<image-name>:<tag>
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

> **Note:** The `progress-deadline` of 30 minutes is important — the first pull
> of large model images can exceed the default 10-minute KNative timeout.

## File structure

```
.
├── build.sh                   # Automated workflow (download, build, archive, save, rehash, etc.)
├── compose.yaml               # Builds and runs the downloader in one command
├── Containerfile.download     # Downloader image (Python + huggingface-hub)
├── download_model.py          # Download script (configurable via env vars)
├── models/                    # Downloaded weights (gitignored except .gitkeep)
│   └── .gitkeep
├── models_archive/            # Archived weights per model (gitignored)
├── save/                      # Exported split tarballs for air-gapped transfer
│   └── <model-slug>/
│       ├── model.tar.part00
│       ├── model.tar.part01
│       ├── ...
│       ├── checksums.sha256
│       ├── checksums.b2       # If b2sum was available at save time
│       └── load.sh            # Verify, assemble, and load helper
├── .gitignore
└── README.md
```

## References

- [RHOAI: Storing a model in an OCI image](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.22/html/serving_models/serving-large-models_serving-large-models#storing-a-model-in-oci-image_serving-large-models)
- [Build and deploy a ModelCar container in OpenShift AI](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)
