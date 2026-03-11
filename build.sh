#!/bin/bash
set -euo pipefail
trap 'echo ""; echo "==> Interrupted."; exit 130' INT

# -- Configuration ----------------------------------------------
MODEL_REPO="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
HF_TOKEN=""
SPLIT_SIZE="4G"
# ---------------------------------------------------------------

ARCHIVE_DIR="models_archive"

# Derive image name: lowercase the repo (OCI spec requires lowercase)
IMAGE_NAME=$(echo "$MODEL_REPO" | tr '[:upper:]' '[:lower:]')

# Derive a filesystem-safe name for archiving: Qwen/Qwen3-VL-30B -> qwen--qwen3-vl-30b
MODEL_SLUG=$(echo "$IMAGE_NAME" | sed 's|/|--|g')

DATE_TAG=$(date +%Y%m%d)
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_NAME}:${DATE_TAG}}"
TOKEN_STATUS=$([ -n "$HF_TOKEN" ] && echo "set" || echo "NOT SET")

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

  # Count total parts
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
      # Assemble results in sorted order
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
ModelCar Builder - package HuggingFace models as OCI images for RHOAI

Usage: ./build.sh <command>

Commands:
  all       Run the full pipeline (download -> build -> archive)
  download  Download model weights into models/
  build     Build the ModelCar OCI image from models/
  archive   Move models/ into models_archive/<model-slug>/
  restore   Copy weights from models_archive/ back into models/
            Optionally pass a path: ./build.sh restore models_archive/<dir>
  save      Save the image as split tar files for air-gapped transfer
            Optionally pass an image tag: ./build.sh save <image:tag>
  rehash    Regenerate checksums for a save directory
            Optionally pass a path: ./build.sh rehash save/<dir>
  clean     Delete all downloaded weights from models/
  status    Show current configuration and state

Configuration:
  Edit MODEL_REPO, HF_TOKEN, and SPLIT_SIZE at the top of this script.

  MODEL_REPO  $MODEL_REPO
  HF_TOKEN    $TOKEN_STATUS
  IMAGE_TAG   $IMAGE_TAG
  SPLIT_SIZE  $SPLIT_SIZE

Override the image tag:
  IMAGE_TAG=my-model:v1 ./build.sh all
EOF
}

cmd_download() {
  if podman image exists modelcar-downloader:latest 2>/dev/null; then
    echo "==> Downloader image already exists, skipping build."
  else
    echo "==> Building downloader image..."
    podman build -f Containerfile.download -t modelcar-downloader:latest .
  fi

  echo ""
  echo "==> Downloading model weights (token: $TOKEN_STATUS)..."
  podman run --rm -it \
    -v "$(pwd)/models:/models:Z" \
    -e "MODEL_REPO=$MODEL_REPO" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "HF_HUB_ENABLE_HF_TRANSFER=1" \
    -e "PYTHONUNBUFFERED=1" \
    modelcar-downloader:latest
}

cmd_build() {
  echo "==> Generating Containerfile..."

  {
    echo "FROM registry.access.redhat.com/ubi9/ubi-micro:latest"
    echo ""
    echo "# Config and metadata files (small, single layer)"
    echo "COPY --chown=0:0 --chmod=555 models/*.json models/*.txt models/*.py /models/"
    echo ""
    echo "# Each safetensor shard as its own layer for parallel/resumable pulls"

    find models -maxdepth 1 -name '*.safetensors' -printf '%f\n' | sort | while read -r shard; do
      echo "COPY --chown=0:0 --chmod=555 models/$shard /models/$shard"
    done

    echo ""
    echo "# nobody user"
    echo "USER 65534"
  } > Containerfile

  echo ""
  echo "-- Generated Containerfile ---------------------------------"
  cat Containerfile
  echo "------------------------------------------------------------"
  echo ""

  echo "==> Building ModelCar image: $IMAGE_TAG"
  podman build --format=oci -t "$IMAGE_TAG" .

  rm -f Containerfile
}

cmd_archive() {
  echo "==> Archiving weights to $ARCHIVE_DIR/$MODEL_SLUG/"
  mkdir -p "$ARCHIVE_DIR/$MODEL_SLUG"
  mv models/* "$ARCHIVE_DIR/$MODEL_SLUG/" 2>/dev/null || true
  touch models/.gitkeep
}

cmd_restore() {
  local source_dir="${1:-$ARCHIVE_DIR/$MODEL_SLUG}"

  if [ ! -d "$source_dir" ]; then
    echo "==> Error: Directory not found: $source_dir/"
    echo ""
    echo "    Available archives:"
    for dir in "$ARCHIVE_DIR"/*/; do
      [ -d "$dir" ] && echo "      $dir"
    done
    exit 1
  fi

  if [ -n "$(find models -mindepth 1 -not -name '.gitkeep' 2>/dev/null)" ]; then
    echo "==> Error: models/ is not empty."
    echo "    Run './build.sh clean' or './build.sh archive' first."
    exit 1
  fi

  echo "==> Restoring weights from $source_dir/ to models/"
  mv "$source_dir"/* models/
  rmdir "$source_dir"
  echo "==> Done. $(find models -type f -not -name '.gitkeep' | wc -l) files restored."
}

cmd_save() {
  local save_tag="${1:-$IMAGE_TAG}"
  local save_slug
  save_slug=$(echo "$save_tag" | tr '[:upper:]' '[:lower:]' | sed 's|[/:]|--|g')
  local output_dir="save/${save_slug}"
  mkdir -p "$output_dir"

  echo "==> Saving image: $save_tag"
  echo "==> Split size:   $SPLIT_SIZE"
  echo "==> Output:       $output_dir/"
  echo ""

  # Get image size for progress bar
  local image_size
  image_size=$(podman image inspect "$save_tag" --format '{{.Size}}' 2>/dev/null || echo "0")

  if command -v pv &>/dev/null && [ "$image_size" -gt 0 ]; then
    podman save "$save_tag" | pv -s "$image_size" | split -b "$SPLIT_SIZE" -d - "${output_dir}/model.tar.part"
  else
    if ! command -v pv &>/dev/null; then
      echo "    (install 'pv' for a progress bar: apt install pv)"
    fi
    podman save "$save_tag" | split -b "$SPLIT_SIZE" -d - "${output_dir}/model.tar.part"
  fi

  echo "==> Generating checksums..."
  generate_checksums "$output_dir"

  # Write a reassembly script
  cat > "${output_dir}/load.sh" <<'LOAD'
#!/bin/bash
set -euo pipefail

verify() {
  # Detect checksum file and matching tool
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

  # Hash all parts in parallel and store results
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

  # Wait for all hashing to complete
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "" >&2

  # Compare results
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
      echo "==> Reassembling and loading image..."
      pipe_with_progress | podman load
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
    echo "  load      Check checksums and load into podman"
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
}

cmd_rehash() {
  local target="${1:-}"

  rehash_dir() {
    local dir="$1"
    echo "==> Regenerating checksums in $dir/"
    generate_checksums "$dir"
  }

  if [ -n "$target" ]; then
    # Rehash a specific directory
    if [ ! -d "$target" ] || ! ls "$target"/model.tar.part* &>/dev/null; then
      echo "==> Error: No parts found in $target/"
      exit 1
    fi
    rehash_dir "$target"
  else
    # Rehash all save directories
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
  echo "==> Cleaning models directory..."
  find models -mindepth 1 -not -name '.gitkeep' -delete 2>/dev/null || true
  echo "==> Done."
}

cmd_status() {
  echo "==> Configuration"
  echo "    MODEL_REPO:  $MODEL_REPO"
  echo "    HF_TOKEN:    $TOKEN_STATUS"
  echo "    IMAGE_TAG:   $IMAGE_TAG"
  echo "    MODEL_SLUG:  $MODEL_SLUG"
  echo "    SPLIT_SIZE:  $SPLIT_SIZE"
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

  echo "==> models_archive/"
  if [ -d "$ARCHIVE_DIR" ] && [ -n "$(ls -A "$ARCHIVE_DIR" 2>/dev/null)" ]; then
    for dir in "$ARCHIVE_DIR"/*/; do
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
  echo "==> Model:   $MODEL_REPO"
  echo "==> Image:   $IMAGE_TAG"
  echo "==> Token:   $TOKEN_STATUS"
  echo ""

  cmd_download

  echo ""
  cmd_build

  echo ""
  cmd_archive

  echo ""
  echo "==> Done."
  echo "    Image:   $IMAGE_TAG"
  echo "    Weights: $ARCHIVE_DIR/$MODEL_SLUG/"
  echo ""
  echo "    Next steps:"
  echo "      podman tag $IMAGE_TAG <your-registry>/$IMAGE_TAG"
  echo "      podman push <your-registry>/$IMAGE_TAG"
  echo ""
  echo "    For air-gapped transfer:"
  echo "      ./build.sh save"
}

# -- Main -------------------------------------------------------

if [ -z "$MODEL_REPO" ]; then
  echo "Error: MODEL_REPO is not set."
  exit 1
fi

case "${1:-}" in
  all)      cmd_all      ;;
  download) cmd_download ;;
  build)    cmd_build    ;;
  archive)  cmd_archive  ;;
  restore)  cmd_restore "${2:-}" ;;
  save)     cmd_save "${2:-}" ;;
  rehash)   cmd_rehash "${2:-}" ;;
  clean)    cmd_clean    ;;
  status)   cmd_status   ;;
  *)        show_help    ;;
esac