from huggingface_hub import snapshot_download
import os

repo_id = os.environ.get("REPO_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
output_dir = os.environ.get("OUTPUT_DIR", "/models")
repo_type = os.environ.get("REPO_TYPE", "model")

ignore = []
if repo_type == "model":
    ignore = ["*.gguf", "*.bin", "*.onnx", "*.msgpack", "*.h5", "*.ot"]

print(f"Downloading {repo_type}: {repo_id} to {output_dir}...")

snapshot_download(
    repo_id=repo_id,
    local_dir=output_dir,
    repo_type=repo_type,
    ignore_patterns=ignore if ignore else None,
)

print("Download complete.")