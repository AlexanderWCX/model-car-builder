from huggingface_hub import snapshot_download
import os

model_repo = os.environ.get("MODEL_REPO", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
output_dir = os.environ.get("OUTPUT_DIR", "/models")

print(f"Downloading {model_repo} to {output_dir}...")

snapshot_download(
    repo_id=model_repo,
    local_dir=output_dir,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
)

print("Download complete.")
