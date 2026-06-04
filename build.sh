#!/bin/bash
set -euo pipefail
trap 'echo ""; echo "==> Interrupted."; exit 130' INT

# -- Configuration ----------------------------------------------
# Set ONE of these. Do not set both.
MODEL_REPO=""
DATASET_REPO=""
HF_TOKEN=""
SPLIT_SIZE="4G"
# Max size per safetensors shard (each shard becomes one OCI layer). Keeps layers
# under the registry's per-layer cap (e.g. Quay MAXIMUM_LAYER_SIZE, default 20G)
# and keeps parallel pulls effective. Set empty or "0" to disable. Units: G/M.
MAX_SHARD_SIZE="4G"
# ---------------------------------------------------------------

# -- Mode detection ---------------------------------------------
if [ -n "$MODEL_REPO" ] && [ -n "$DATASET_REPO" ]; then
  echo "Error: Both MODEL_REPO and DATASET_REPO are set. Set only one at a time."
  exit 1
fi

if [ -n "$MODEL_REPO" ]; then
  MODE="model"
  REPO_ID="$MODEL_REPO"
  WORK_DIR="models"
  ARCHIVE_DIR="models_archive"
elif [ -n "$DATASET_REPO" ]; then
  MODE="dataset"
  REPO_ID="$DATASET_REPO"
  WORK_DIR="datasets"
  ARCHIVE_DIR="datasets_archive"
else
  MODE="none"
  REPO_ID=""
  WORK_DIR=""
  ARCHIVE_DIR=""
fi

# Derived variables (only when a repo is set)
if [ -n "$REPO_ID" ]; then
  IMAGE_NAME=$(echo "$REPO_ID" | tr '[:upper:]' '[:lower:]')
  SLUG=$(echo "$IMAGE_NAME" | sed 's|/|--|g')
  DATE_TAG=$(date +%Y%m%d)
  IMAGE_TAG="${IMAGE_TAG:-${IMAGE_NAME}:${DATE_TAG}}"
else
  IMAGE_NAME=""
  SLUG=""
  DATE_TAG=$(date +%Y%m%d)
  IMAGE_TAG=""
fi

TOKEN_STATUS=$([ -n "$HF_TOKEN" ] && echo "set" || echo "NOT SET")

# Detect container runtime (prefer podman)
detect_runtime() {
  if command -v podman &>/dev/null; then
    echo "podman"
  elif command -v docker &>/dev/null; then
    echo "docker"
  else
    echo "==> Error: Neither podman nor docker found." >&2
    exit 1
  fi
}
RUNTIME=$(detect_runtime)

# -- Helpers ----------------------------------------------------

require_repo() {
  if [ "$MODE" = "none" ]; then
    echo "==> Error: Neither MODEL_REPO nor DATASET_REPO is set."
    echo "    Edit the top of build.sh and set one of them."
    exit 1
  fi
}

# Generate checksums for all parts in a directory, per-file in parallel
generate_checksums() {
  local dir="$1"
  local tmpdir
  tmpdir=$(mktemp -d)
  trap "rm -rf $tmpdir" RETURN

  local hash_cmds=()
  if command -v b2sum &>/dev/null; then
    hash_cmds+=("b2sum:checksums.b2")
    echo "    BLAKE2 (b2sum)..." >&2
  fi
  hash_cmds+=("sha256sum:checksums.sha256")
  echo "    SHA-256 (sha256sum)..." >&2

  local total_parts
  total_parts=$(ls "$dir"/model.tar.part* 2>/dev/null | wc -l)
  local total_jobs=$(( total_parts * ${#hash_cmds[@]} ))

  local pids=()
  for entry in "${hash_cmds[@]}"; do
    local cmd="${entry%%:*}"
    local outfile="${entry##*:}"
    (
      cd "$dir"
      local file_pids=()
      for part in model.tar.part*; do
        (
          hash=$("$cmd" "$part" | awk '{print $1}')
          echo "$hash  $part" > "$tmpdir/${cmd}_${part}"
        ) &
        file_pids+=($!)
      done
      for pid in "${file_pids[@]}"; do
        wait "$pid"
      done
      cat "$tmpdir"/${cmd}_model.tar.part* | sort -k2 > "$outfile"
    ) &
    pids+=($!)
  done

  # Progress monitor
  while true; do
    local done_count
    done_count=$(find "$tmpdir" -type f 2>/dev/null | wc -l)
    printf "\r    Progress: %d/%d hashes complete" "$done_count" "$total_jobs" >&2
    if [ "$done_count" -ge "$total_jobs" ]; then
      break
    fi
    sleep 1
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  printf "\r    Progress: %d/%d hashes complete\n" "$total_jobs" "$total_jobs" >&2
  echo "    Done." >&2
}

# -- Commands ---------------------------------------------------

show_help() {
  cat <<EOF
ModelCar Builder - package HuggingFace models and datasets as OCI images

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

Configuration:
  Set ONE of MODEL_REPO or DATASET_REPO at the top of this script.

  MODE          $MODE
  MODEL_REPO    ${MODEL_REPO:-(not set)}
  DATASET_REPO  ${DATASET_REPO:-(not set)}
  HF_TOKEN      $TOKEN_STATUS
  IMAGE_TAG     ${IMAGE_TAG:-(not set)}
  SPLIT_SIZE    $SPLIT_SIZE
  MAX_SHARD_SIZE ${MAX_SHARD_SIZE:-(disabled)}
  RUNTIME       $RUNTIME

Override the image tag:
  IMAGE_TAG=my-model:v1 ./build.sh all
EOF
}

cmd_download() {
  require_repo

  if [ -n "$(find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "==> Warning: $WORK_DIR/ is not empty."
    read -rp "    Resume previous download? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
      echo "    Aborting. Run './build.sh clean' to start fresh."
      exit 1
    fi
  fi

  if $RUNTIME image exists modelcar-downloader:latest 2>/dev/null || $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
    echo "==> Downloader image already exists, skipping build."
  else
    echo "==> Building downloader image..."
    $RUNTIME build -f Containerfile.download -t modelcar-downloader:latest .
  fi

  local vol_suffix=""
  [ "$RUNTIME" = "podman" ] && vol_suffix=":Z"

  local repo_type="model"
  local output_dir="/models"
  if [ "$MODE" = "dataset" ]; then
    repo_type="dataset"
    output_dir="/datasets"
  fi

  echo ""
  echo "==> Downloading $MODE: $REPO_ID (token: $TOKEN_STATUS)..."
  $RUNTIME run --rm -it \
    -v "$(pwd)/$WORK_DIR:${output_dir}${vol_suffix}" \
    -e "REPO_ID=$REPO_ID" \
    -e "OUTPUT_DIR=$output_dir" \
    -e "REPO_TYPE=$repo_type" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "HF_HUB_ENABLE_HF_TRANSFER=1" \
    -e "PYTHONUNBUFFERED=1" \
    modelcar-downloader:latest
}

cmd_convert() {
  if [ "$MODE" != "dataset" ]; then
    echo "==> Error: Convert is only for datasets."
    exit 1
  fi

  if [ -z "$(find "$WORK_DIR" -name '*.parquet' 2>/dev/null)" ]; then
    echo "==> No parquet files found in $WORK_DIR/, skipping conversion."
    return 0
  fi

  if $RUNTIME image exists modelcar-downloader:latest 2>/dev/null || $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
    : # image exists
  else
    echo "==> Building downloader image..."
    $RUNTIME build -f Containerfile.download -t modelcar-downloader:latest .
  fi

  local vol_suffix=""
  [ "$RUNTIME" = "podman" ] && vol_suffix=":Z"

  echo "==> Converting parquet files to JSONL..."
  $RUNTIME run --rm -it \
    -v "$(pwd)/$WORK_DIR:/datasets${vol_suffix}" \
    -e "INPUT_DIR=/datasets" \
    -e "OUTPUT_DIR=/datasets" \
    -e "REMOVE_PARQUET=1" \
    -e "PYTHONUNBUFFERED=1" \
    --entrypoint python \
    modelcar-downloader:latest \
    /app/convert_parquet.py
}

cmd_reshard() {
  require_repo

  if [ "$MODE" != "model" ]; then
    echo "==> Reshard is only for models, skipping."
    return 0
  fi

  local has_staging=0
  [ -d "$WORK_DIR/.reshard-staging" ] && has_staging=1

  local shard_size="${MAX_SHARD_SIZE:-0}"
  [ -z "$shard_size" ] && shard_size=0

  # Skip only when resharding is disabled AND there is nothing to recover. An
  # interrupted reshard must always be resolved (it may have left models/ in a
  # half-committed state), so a leftover staging dir forces the container to run
  # even with resharding disabled.
  if [ "$shard_size" = "0" ] && [ "$has_staging" -eq 0 ]; then
    echo "==> MAX_SHARD_SIZE disabled, skipping reshard."
    return 0
  fi

  if [ -z "$(find "$WORK_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null)" ] && [ "$has_staging" -eq 0 ]; then
    echo "==> No safetensors in $WORK_DIR/, skipping reshard."
    return 0
  fi

  if $RUNTIME image exists modelcar-downloader:latest 2>/dev/null || $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
    : # image exists
  else
    echo "==> Building downloader image..."
    $RUNTIME build -f Containerfile.download -t modelcar-downloader:latest .
  fi

  local vol_suffix=""
  [ "$RUNTIME" = "podman" ] && vol_suffix=":Z"

  if [ "$shard_size" = "0" ]; then
    echo "==> Recovering interrupted reshard (resharding disabled)..."
  else
    echo "==> Resharding safetensors to <= $shard_size per shard..."
  fi
  $RUNTIME run --rm \
    -v "$(pwd)/$WORK_DIR:/models${vol_suffix}" \
    -e "MODELS_DIR=/models" \
    -e "MAX_SHARD_SIZE=$shard_size" \
    -e "PYTHONUNBUFFERED=1" \
    --entrypoint python \
    modelcar-downloader:latest \
    /app/reshard_safetensors.py
}

cmd_build() {
  require_repo

  local single_layer=0
  if [ "${1:-}" = "--single-layer" ]; then
    single_layer=1
    IMAGE_TAG="${IMAGE_TAG%-single}-single"
    echo "==> Generating Containerfile ($MODE, single layer)..."
  else
    echo "==> Generating Containerfile ($MODE)..."
  fi

  # Re-shard oversized safetensors before building so no single layer exceeds the
  # registry's per-layer limit. --single-layer is one layer by design so it never
  # splits, but it must still resolve a half-committed reshard left by a prior
  # interrupted run (otherwise that partial state gets baked into the image).
  if [ "$MODE" = "model" ]; then
    if [ "$single_layer" -eq 0 ]; then
      cmd_reshard
      echo ""
    elif [ -d "$WORK_DIR/.reshard-staging" ]; then
      echo "==> Resolving an interrupted reshard before the single-layer build..."
      ( MAX_SHARD_SIZE=0; cmd_reshard )
      echo ""
    fi
  fi

  if [ "$MODE" = "model" ]; then
    _build_model "$single_layer"
  else
    _build_dataset
  fi

  echo ""
  echo "-- Generated Containerfile ---------------------------------"
  cat Containerfile
  echo "------------------------------------------------------------"
  echo ""

  echo "==> Building image: $IMAGE_TAG"
  if [ "$RUNTIME" = "podman" ]; then
    $RUNTIME build --format=oci -f Containerfile -t "$IMAGE_TAG" .
  else
    $RUNTIME build -f Containerfile -t "$IMAGE_TAG" .
  fi

  rm -f Containerfile

  # Build tokenizer image for models
  if [ "$MODE" = "model" ]; then
    _build_tokenizer
  fi
}

_build_model() {
  local single_layer="$1"

  {
    echo "FROM registry.access.redhat.com/ubi9/ubi-micro:latest"
    echo ""

    if [ "$single_layer" -eq 1 ]; then
      echo "# All model files in a single layer"
      echo "COPY --chown=0:0 --chmod=555 models/ /models/"
    else
      # Config and metadata files (top-level, non-safetensor)
      local meta_files
      meta_files=$(find models -maxdepth 1 -type f -not -name '*.safetensors' -not -name '.gitkeep' -not -name '*.metadata' -not -name '*.lock' 2>/dev/null)
      if [ -n "$meta_files" ]; then
        echo "# Config and metadata files (small, single layer)"
        local globs=""
        for f in $meta_files; do
          globs="$globs $f"
        done
        echo "COPY --chown=0:0 --chmod=555$globs /models/"
        echo ""
      fi

      # Subdirectories (e.g. 1_Pooling/, 2_Dense/)
      local subdirs
      subdirs=$(find models -mindepth 1 -maxdepth 1 -type d -not -name '.cache' -not -name 'download' -not -name '.reshard-staging' 2>/dev/null)
      if [ -n "$subdirs" ]; then
        echo "# Model subdirectories"
        for dir in $subdirs; do
          local dirname
          dirname=$(basename "$dir")
          echo "COPY --chown=0:0 --chmod=555 models/$dirname/ /models/$dirname/"
        done
        echo ""
      fi

      # Each safetensor shard as its own layer
      local shards
      shards=$(find models -maxdepth 1 -name '*.safetensors' -printf '%f\n' | sort)
      if [ -n "$shards" ]; then
        echo "# Each safetensor shard as its own layer for parallel/resumable pulls"
        echo "$shards" | while read -r shard; do
          echo "COPY --chown=0:0 --chmod=555 models/$shard /models/$shard"
        done
        echo ""
      fi
    fi

    echo "# nobody user"
    echo "USER 65534"
  } > Containerfile
}

_build_dataset() {
  {
    echo "FROM registry.access.redhat.com/ubi9/ubi-micro:latest"
    echo ""

    # Top-level small files (not parquet/jsonl)
    local top_small
    top_small=$(find datasets -maxdepth 1 -type f -not -name '*.parquet' -not -name '*.jsonl' -not -name '.gitkeep' -not -name '*.metadata' -not -name '*.lock' 2>/dev/null)
    if [ -n "$top_small" ]; then
      echo "# Dataset metadata"
      local flist=""
      for f in $top_small; do
        flist="$flist $f"
      done
      echo "COPY --chown=0:0 --chmod=555$flist /datasets/"
      echo ""
    fi

    # Subdirectories
    local subdirs
    subdirs=$(find datasets -mindepth 1 -maxdepth 1 -type d -not -name '.cache' -not -name 'download' 2>/dev/null)
    if [ -n "$subdirs" ]; then
      for dir in $subdirs; do
        local dirname
        dirname=$(basename "$dir")

        # Small files in subdirectory
        local other_files
        other_files=$(find "$dir" -maxdepth 1 -type f -not -name '*.parquet' -not -name '*.jsonl' -not -name '*.metadata' -not -name '*.lock' 2>/dev/null)
        if [ -n "$other_files" ]; then
          echo "# $dirname/ metadata"
          local ofiles=""
          for f in $other_files; do
            ofiles="$ofiles $f"
          done
          echo "COPY --chown=0:0 --chmod=555$ofiles /datasets/$dirname/"
          echo ""
        fi

        # Large data files in subdirectory (jsonl or parquet, one layer each)
        local data_files
        data_files=$(find "$dir" -maxdepth 1 \( -name '*.jsonl' -o -name '*.parquet' \) -printf '%f\n' | sort)
        if [ -n "$data_files" ]; then
          echo "# $dirname/ data shards (one layer each)"
          echo "$data_files" | while read -r df; do
            echo "COPY --chown=0:0 --chmod=555 datasets/$dirname/$df /datasets/$dirname/$df"
          done
          echo ""
        fi
      done
    fi

    # Top-level large data files (jsonl or parquet, one layer each)
    local top_data
    top_data=$(find datasets -maxdepth 1 \( -name '*.jsonl' -o -name '*.parquet' \) -printf '%f\n' | sort)
    if [ -n "$top_data" ]; then
      echo "# Data shards (one layer each)"
      echo "$top_data" | while read -r df; do
        echo "COPY --chown=0:0 --chmod=555 datasets/$df /datasets/$df"
      done
      echo ""
    fi

    echo "# nobody user"
    echo "USER 65534"
  } > Containerfile
}

_build_tokenizer() {
  local tokenizer_tag="${IMAGE_TAG}-tokenizer"
  local tokenizer_files=""
  for f in models/tokenizer.json models/tokenizer_config.json models/config.json; do
    if [ -f "$f" ]; then
      tokenizer_files="$tokenizer_files $f"
    fi
  done

  if [ -n "$tokenizer_files" ]; then
    echo ""
    echo "==> Building tokenizer image: $tokenizer_tag"
    {
      echo "FROM registry.access.redhat.com/ubi9/ubi-micro:latest"
      echo ""
      echo "# Tokenizer files only"
      echo "COPY --chown=0:0 --chmod=555$tokenizer_files /models/"
      echo ""
      echo "# nobody user"
      echo "USER 65534"
    } > Containerfile

    if [ "$RUNTIME" = "podman" ]; then
      $RUNTIME build --format=oci -f Containerfile -t "$tokenizer_tag" .
    else
      $RUNTIME build -f Containerfile -t "$tokenizer_tag" .
    fi

    rm -f Containerfile
  else
    echo ""
    echo "==> Warning: No tokenizer files found, skipping tokenizer image."
  fi
}

cmd_archive() {
  require_repo

  echo "==> Archiving $WORK_DIR/ to $ARCHIVE_DIR/$SLUG/"
  mkdir -p "$ARCHIVE_DIR/$SLUG"

  # Move ALL model content -- including hidden files like .gitattributes and
  # .eval_results, which the build copies into the image. A plain `mv $WORK_DIR/*`
  # glob skips dotfiles and would leave them behind, producing an incomplete
  # archive. Keep the placeholder and the transient HF cache / reshard staging
  # out of the archive. (Best-effort host move first; root-owned dirs fail here.)
  find "$WORK_DIR" -mindepth 1 -maxdepth 1 \
    -not -name '.gitkeep' \
    -not -name '.cache' \
    -not -name '.reshard-staging' \
    -exec mv -t "$ARCHIVE_DIR/$SLUG/" {} + 2>/dev/null || true
  find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' -delete 2>/dev/null || true

  # The downloader container writes as root, so root-owned directories (e.g.
  # .eval_results) can't be moved -- and .cache can't be deleted -- by the host
  # user. If anything but .gitkeep remains, finish the move + cleanup in a root
  # container (whole repo mounted so moves stay on one filesystem = fast renames).
  if [ -n "$(find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    if $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
      echo "==> Finishing archive of root-owned content via container..."
      $RUNTIME run --rm -v "$(pwd):/repo" --entrypoint sh modelcar-downloader:latest -c \
        "cd /repo && find '$WORK_DIR' -mindepth 1 -maxdepth 1 -not -name .gitkeep -not -name .cache -not -name .reshard-staging -exec mv -t '$ARCHIVE_DIR/$SLUG/' {} + && find '$WORK_DIR' -mindepth 1 -not -name .gitkeep -delete"
    else
      echo "==> Warning: root-owned content remains in $WORK_DIR/ and the downloader"
      echo "    image is unavailable to move it. Rebuild it (or use sudo), then re-run archive."
    fi
  fi

  touch "$WORK_DIR/.gitkeep"
}

cmd_restore() {
  require_repo

  local source_dir="${1:-$ARCHIVE_DIR/$SLUG}"

  if [ ! -d "$source_dir" ]; then
    echo "==> Error: Directory not found: $source_dir/"
    echo ""
    echo "    Available archives:"
    for dir in "$ARCHIVE_DIR"/*/; do
      [ -d "$dir" ] && echo "      $dir"
    done
    exit 1
  fi

  if [ -n "$(find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "==> Error: $WORK_DIR/ is not empty."
    echo "    Run './build.sh clean' first."
    exit 1
  fi

  echo "==> Restoring from $source_dir/ to $WORK_DIR/"
  # Move everything back, including hidden model content (.gitattributes,
  # .eval_results) -- a `mv $source_dir/*` glob skips dotfiles, which would both
  # lose them and leave the source dir non-empty so the rmdir below fails.
  # Best-effort host move first; root-owned dirs need a root container (below).
  find "$source_dir" -mindepth 1 -maxdepth 1 -exec mv -t "$WORK_DIR/" {} + 2>/dev/null || true
  if [ -n "$(find "$source_dir" -mindepth 1 2>/dev/null)" ]; then
    case "$source_dir" in
      /*) : ;;  # absolute/external path: cannot be mapped into the repo mount
      *)
        if $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
          echo "==> Restoring root-owned content via container..."
          $RUNTIME run --rm -v "$(pwd):/repo" --entrypoint sh modelcar-downloader:latest -c \
            "cd /repo && find '$source_dir' -mindepth 1 -maxdepth 1 -exec mv -t '$WORK_DIR/' {} +"
        fi
        ;;
    esac
  fi
  rmdir "$source_dir" 2>/dev/null || \
    echo "==> Note: $source_dir/ not removed (root-owned leftovers?); files were restored."
  echo "==> Done. $(find "$WORK_DIR" -type f -not -name '.gitkeep' | wc -l) files restored."
}

cmd_save() {
  local save_tag=""
  local save_runtime="$RUNTIME"

  while [ $# -gt 0 ]; do
    case "$1" in
      --docker)
        if ! command -v docker &>/dev/null; then
          echo "==> Error: --docker requested but docker is not installed."
          exit 1
        fi
        save_runtime="docker"
        ;;
      *)
        if [ -z "$save_tag" ]; then
          save_tag="$1"
        else
          echo "==> Error: Unexpected argument: $1"
          echo "    Usage: ./build.sh save [--docker] [<image:tag>]"
          exit 1
        fi
        ;;
    esac
    shift
  done

  save_tag="${save_tag:-$IMAGE_TAG}"

  if [ -z "$save_tag" ]; then
    echo "==> Error: No image tag specified and none derived from config."
    echo "    Usage: ./build.sh save [--docker] <image:tag>"
    exit 1
  fi

  local save_slug
  save_slug=$(echo "$save_tag" | tr '[:upper:]' '[:lower:]' | sed 's|[/:]|--|g')
  local output_dir="save/${save_slug}"

  if [ -d "$output_dir" ] && ls "$output_dir"/model.tar.part* &>/dev/null; then
    echo "==> Error: Save directory already exists: $output_dir/"
    echo "    Delete it manually if you want to re-save."
    exit 1
  fi

  mkdir -p "$output_dir"

  echo "==> Saving image: $save_tag"
  echo "==> Split size:   $SPLIT_SIZE"
  echo "==> Runtime:      $save_runtime"
  echo "==> Output:       $output_dir/"
  echo ""

  local save_cmd
  if [ "$save_runtime" = "podman" ]; then
    save_cmd="$save_runtime save --format=docker-archive $save_tag"
  else
    save_cmd="$save_runtime save $save_tag"
  fi

  local image_size
  image_size=$($save_runtime image inspect "$save_tag" --format '{{.Size}}' 2>/dev/null || echo "0")

  if command -v pv &>/dev/null && [ "$image_size" -gt 0 ]; then
    $save_cmd | pv -s "$image_size" | split -b "$SPLIT_SIZE" -d - "${output_dir}/model.tar.part"
  else
    if ! command -v pv &>/dev/null; then
      echo "    (install 'pv' for a progress bar: apt install pv / dnf install pv)"
    fi
    $save_cmd | split -b "$SPLIT_SIZE" -d - "${output_dir}/model.tar.part"
  fi

  echo "==> Generating checksums..."
  generate_checksums "$output_dir"

  # Write a reassembly script
  cat > "${output_dir}/load.sh" <<'LOAD'
#!/bin/bash
set -euo pipefail

# Auto-detect container runtime
if command -v podman &>/dev/null; then
  RUNTIME="podman"
elif command -v docker &>/dev/null; then
  RUNTIME="docker"
else
  echo "==> Error: Neither podman nor docker found."
  exit 1
fi

verify() {
  local checksum_file hash_cmd
  if [ -f checksums.b2 ] && command -v b2sum &>/dev/null; then
    checksum_file="checksums.b2"
    hash_cmd="b2sum"
  elif [ -f checksums.sha256 ] && command -v sha256sum &>/dev/null; then
    checksum_file="checksums.sha256"
    hash_cmd="sha256sum"
  elif [ -f checksums.b2 ]; then
    echo "==> Error: checksums.b2 found but b2sum is not installed."
    echo "    Install b2sum (part of coreutils) or provide checksums.sha256."
    exit 1
  else
    echo "==> Error: No checksum file found (expected checksums.b2 or checksums.sha256)."
    exit 1
  fi

  local total_parts
  total_parts=$(wc -l < "$checksum_file")

  echo "==> Verifying checksums ($hash_cmd, $total_parts parts)..."
  echo ""

  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN

  local pids=()
  while IFS= read -r line; do
    file=$(echo "$line" | awk '{print $2}')
    (
      hash=$("$hash_cmd" "$file" | awk '{print $1}')
      echo "$hash" > "$tmpdir/$file"
    ) &
    pids+=($!)
  done < "$checksum_file"

  # Progress monitor
  while true; do
    local done_count
    done_count=$(find "$tmpdir" -type f 2>/dev/null | wc -l)
    printf "\r    Hashing: %d/%d parts complete" "$done_count" "$total_parts" >&2
    if [ "$done_count" -ge "$total_parts" ]; then
      break
    fi
    sleep 1
  done
  printf "\r    Hashing: %d/%d parts complete\n" "$total_parts" "$total_parts" >&2

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "" >&2

  local failed=0
  local total=0

  while IFS= read -r line; do
    file=$(echo "$line" | awk '{print $2}')
    expected=$(echo "$line" | awk '{print $1}')
    total=$((total + 1))

    actual=$(cat "$tmpdir/$file")

    if [ "$expected" = "$actual" ]; then
      echo "    OK   $file"
    else
      echo "    FAIL $file"
      failed=$((failed + 1))
    fi
  done < "$checksum_file"

  echo ""

  if [ "$failed" -gt 0 ]; then
    echo "==> $failed of $total parts FAILED checksum verification."
    echo "    Re-transfer the failed files and run './load.sh verify' again."
    return 1
  else
    echo "==> All $total parts passed checksum verification."
    return 0
  fi
}

pipe_with_progress() {
  local total_size
  total_size=$(du -cb model.tar.part* 2>/dev/null | tail -1 | cut -f1)

  if command -v pv &>/dev/null && [ "$total_size" -gt 0 ]; then
    cat model.tar.part* | pv -s "$total_size"
  else
    if ! command -v pv &>/dev/null; then
      echo "    (install 'pv' for a progress bar: apt install pv / dnf install pv)" >&2
    fi
    cat model.tar.part*
  fi
}

case "${1:-}" in
  verify)
    verify
    ;;
  assemble)
    if verify; then
      echo ""
      echo "==> Reassembling into model.tar..."
      pipe_with_progress > model.tar
      echo "==> Done. Output: model.tar ($(du -h model.tar | cut -f1))"
    else
      exit 1
    fi
    ;;
  load)
    if verify; then
      echo ""
      echo "==> Reassembling and loading image ($RUNTIME)..."
      pipe_with_progress | $RUNTIME load
      echo "==> Done."
    else
      exit 1
    fi
    ;;
  *)
    echo "ModelCar Loader - verify and load split image tarballs"
    echo ""
    echo "Usage: ./load.sh <command>"
    echo ""
    echo "Commands:"
    echo "  verify    Check checksums only"
    echo "  assemble  Check checksums and reassemble into model.tar"
    echo "  load      Check checksums and load into $RUNTIME"
    ;;
esac
LOAD
  chmod +x "${output_dir}/load.sh"

  echo ""
  echo "==> Saved to $output_dir/:"
  ls -lh "$output_dir/"
  echo ""
  echo "    Transfer all files in $output_dir/ to the air-gapped host,"
  echo "    then run ./load.sh to verify and load the image."

  # Auto-save tokenizer image if it exists (model mode only)
  local tokenizer_tag="${save_tag}-tokenizer"
  if $save_runtime image inspect "$tokenizer_tag" &>/dev/null; then
    echo ""
    echo "==> Tokenizer image found, saving: $tokenizer_tag"
    local tokenizer_slug="${save_slug}-tokenizer"
    local tokenizer_dir="save/${tokenizer_slug}"
    mkdir -p "$tokenizer_dir"

    if [ "$save_runtime" = "podman" ]; then
      $save_runtime save --format=docker-archive "$tokenizer_tag" > "${tokenizer_dir}/model.tar.part00"
    else
      $save_runtime save "$tokenizer_tag" > "${tokenizer_dir}/model.tar.part00"
    fi

    echo "==> Generating checksums..."
    generate_checksums "$tokenizer_dir"

    cat > "${tokenizer_dir}/load.sh" <<'TLOAD'
#!/bin/bash
set -euo pipefail
if command -v podman &>/dev/null; then RUNTIME="podman"
elif command -v docker &>/dev/null; then RUNTIME="docker"
else echo "==> Error: Neither podman nor docker found."; exit 1; fi

echo "==> Verifying checksum..."
if [ -f checksums.b2 ] && command -v b2sum &>/dev/null; then
  b2sum -c checksums.b2
elif [ -f checksums.sha256 ]; then
  sha256sum -c checksums.sha256
fi

echo "==> Loading tokenizer image ($RUNTIME)..."
$RUNTIME load -i model.tar.part00
echo "==> Done."
TLOAD
    chmod +x "${tokenizer_dir}/load.sh"

    echo ""
    echo "==> Saved tokenizer to $tokenizer_dir/:"
    ls -lh "$tokenizer_dir/"
  fi
}

cmd_rehash() {
  local target="${1:-}"

  rehash_dir() {
    local dir="$1"
    echo "==> Regenerating checksums in $dir/"
    generate_checksums "$dir"
  }

  if [ -n "$target" ]; then
    if [ ! -d "$target" ] || ! ls "$target"/model.tar.part* &>/dev/null; then
      echo "==> Error: No parts found in $target/"
      exit 1
    fi
    rehash_dir "$target"
  else
    local found=0
    for dir in save/*/; do
      if [ -d "$dir" ] && ls "$dir"/model.tar.part* &>/dev/null; then
        rehash_dir "$dir"
        echo ""
        found=1
      fi
    done
    if [ "$found" -eq 0 ]; then
      echo "==> No save directories found."
    fi
  fi

  echo "==> Done."
}

cmd_clean() {
  echo "==> Cleaning $WORK_DIR/..."
  find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' -delete 2>/dev/null || true

  # Older downloads (run as root) leave root-owned files under .cache/ that the
  # host user can't delete. Detect leftovers and remove them via a root container.
  if [ -n "$(find "$WORK_DIR" -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "==> Removing root-owned leftovers via container..."
    if $RUNTIME image inspect modelcar-downloader:latest &>/dev/null; then
      $RUNTIME run --rm -v "$(pwd)/$WORK_DIR:/work" --entrypoint find \
        modelcar-downloader:latest /work -mindepth 1 -not -name '.gitkeep' -delete
    else
      echo "==> Error: leftovers are root-owned and the downloader image is unavailable."
      echo "    Run 'sudo rm -rf $WORK_DIR/.cache' or rebuild the downloader image, then retry."
      exit 1
    fi
  fi
  echo "==> Done."
}

cmd_status() {
  echo "==> Configuration"
  echo "    MODE:          $MODE"
  echo "    MODEL_REPO:    ${MODEL_REPO:-(not set)}"
  echo "    DATASET_REPO:  ${DATASET_REPO:-(not set)}"
  echo "    HF_TOKEN:      $TOKEN_STATUS"
  echo "    IMAGE_TAG:     ${IMAGE_TAG:-(not set)}"
  echo "    SLUG:          ${SLUG:-(not set)}"
  echo "    SPLIT_SIZE:    $SPLIT_SIZE"
  echo "    MAX_SHARD_SIZE: ${MAX_SHARD_SIZE:-(disabled)}"
  echo "    RUNTIME:       $RUNTIME"
  echo ""

  echo "==> models/"
  if [ -z "$(find models -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "    (empty)"
  else
    local count size
    count=$(find models -type f -not -name '.gitkeep' -not -name '*.metadata' -not -name '*.lock' | wc -l)
    size=$(du -sh models 2>/dev/null | cut -f1)
    echo "    $count files, $size total"
  fi
  echo ""

  echo "==> datasets/"
  if [ -z "$(find datasets -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "    (empty)"
  else
    local count size
    count=$(find datasets -type f -not -name '.gitkeep' -not -name '*.metadata' -not -name '*.lock' | wc -l)
    size=$(du -sh datasets 2>/dev/null | cut -f1)
    echo "    $count files, $size total"
  fi
  echo ""

  echo "==> models_archive/"
  if [ -d "models_archive" ] && [ -n "$(ls -A "models_archive" 2>/dev/null)" ]; then
    for dir in models_archive/*/; do
      local slug size
      slug=$(basename "$dir")
      size=$(du -sh "$dir" 2>/dev/null | cut -f1)
      echo "    $slug ($size)"
    done
  else
    echo "    (empty)"
  fi
  echo ""

  echo "==> datasets_archive/"
  if [ -d "datasets_archive" ] && [ -n "$(ls -A "datasets_archive" 2>/dev/null)" ]; then
    for dir in datasets_archive/*/; do
      local slug size
      slug=$(basename "$dir")
      size=$(du -sh "$dir" 2>/dev/null | cut -f1)
      echo "    $slug ($size)"
    done
  else
    echo "    (empty)"
  fi
  echo ""

  echo "==> save/"
  if [ -d "save" ] && [ -n "$(ls -A "save" 2>/dev/null)" ]; then
    for dir in save/*/; do
      local slug count size
      slug=$(basename "$dir")
      count=$(find "$dir" -name 'model.tar.part*' | wc -l)
      size=$(du -sh "$dir" 2>/dev/null | cut -f1)
      echo "    $slug ($count parts, $size total)"
    done
  else
    echo "    (empty)"
  fi
}

cmd_all() {
  require_repo

  echo "==> Mode:    $MODE"
  echo "==> Repo:    $REPO_ID"
  echo "==> Image:   $IMAGE_TAG"
  echo "==> Token:   $TOKEN_STATUS"
  echo "==> Runtime: $RUNTIME"
  echo ""

  cmd_download

  if [ "$MODE" = "dataset" ]; then
    echo ""
    cmd_convert
  fi

  echo ""
  cmd_build

  echo ""
  cmd_archive

  echo ""
  echo "==> Done."
  echo "    Image:   $IMAGE_TAG"
  if [ "$MODE" = "model" ]; then
    echo "    Tokenizer: ${IMAGE_TAG}-tokenizer"
  fi
  echo "    Archive: $ARCHIVE_DIR/$SLUG/"
  echo ""
  echo "    Next steps:"
  echo "      $RUNTIME tag $IMAGE_TAG <your-registry>/$IMAGE_TAG"
  echo "      $RUNTIME push <your-registry>/$IMAGE_TAG"
  echo ""
  echo "    For air-gapped transfer:"
  echo "      ./build.sh save"
}

# -- Main -------------------------------------------------------

case "${1:-}" in
  all)      cmd_all      ;;
  download) cmd_download ;;
  build)    cmd_build "${2:-}" ;;
  reshard)  cmd_reshard  ;;
  archive)  cmd_archive  ;;
  restore)  cmd_restore "${2:-}" ;;
  convert)  cmd_convert  ;;
  save)     shift; cmd_save "$@" ;;
  rehash)   cmd_rehash "${2:-}" ;;
  clean)    cmd_clean    ;;
  status)   cmd_status   ;;
  *)        show_help    ;;
esac