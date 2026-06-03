#!/usr/bin/env python3
"""Re-shard top-level *.safetensors files so no single shard exceeds a target size.

Why: the ModelCar build gives each top-level safetensors file its own OCI layer
(for parallel/resumable pulls). When an upstream repo packs most weights into one
giant file (e.g. a single ~50 GB shard), that becomes one ~50 GB layer -- which
blows past container-registry per-layer limits (Quay's MAXIMUM_LAYER_SIZE
defaults to 20G) and also defeats parallel pulls.

This rewrites the weights into evenly sized shards by repacking at the raw-byte
level -- no torch/numpy, fully dtype-agnostic (bf16/fp8/etc. are copied verbatim)
-- and regenerates model.safetensors.index.json so the build produces many small,
registry-friendly layers.

Properties:
  * Idempotent: if every shard is already <= the target it does nothing.
  * Crash-safe: new shards are written into a staging subdirectory while the
    originals are left untouched; a COMMITTED marker recording the complete final
    filename set is fsync'd into place only after all shard data is fsync'd. The
    commit then MOVES the staged files into place BEFORE deleting any original,
    and decides what to delete from the immutable marker set -- so it is
    re-entrant: an interrupted commit is finished, not corrupted, on the next run,
    and an interrupted write is discarded (originals intact). This matters because
    `build.sh restore` MOVES weights out of the archive, so models/ can be the
    only copy.

Env:
  MODELS_DIR      directory containing top-level *.safetensors (default /models)
  MAX_SHARD_SIZE  target max bytes per shard: "4G", "512M", or a raw byte count.
                  Binary units: K=1024, M=1024^2, G=1024^3 (default "4G").
                  Empty or "0" disables resharding, but staging recovery still runs.

safetensors format (https://github.com/huggingface/safetensors):
  [8 bytes: little-endian u64 header length N][N bytes: JSON header][data buffer]
  Header maps "name" -> {"dtype", "shape", "data_offsets": [begin, end]} where
  offsets are relative to the start of the data buffer (one-past end). A special
  "__metadata__" key holds a free-form string->string map. The header MAY be
  trailing-padded with spaces (0x20).
"""
import json
import os
import shutil
import struct
import sys

COPY_CHUNK = 16 * 1024 * 1024  # streaming copy buffer -> flat memory use
INDEX_NAME = "model.safetensors.index.json"
STAGING_DIR = ".reshard-staging"
COMMIT_MARKER = "COMMITTED"


def parse_size(s):
    s = str(s).strip()
    if not s:
        return 0
    mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    unit = s[-1].upper()
    if unit in mult:
        return int(float(s[:-1]) * mult[unit])
    return int(s)


def human(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{int(n)}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024


def fsync_dir(path):
    """fsync a directory so its entries (creates/renames/unlinks) are durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_header(path):
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too small to be a safetensors file")
        (hlen,) = struct.unpack("<Q", raw)
        hjson = f.read(hlen)
        if len(hjson) != hlen:
            raise ValueError(f"{path}: truncated header (want {hlen}, got {len(hjson)})")
    return json.loads(hjson), 8 + hlen


def write_shard(out_path, src_dir, tensors, metadata):
    """Write one safetensors shard containing `tensors` (in the given order),
    streaming each tensor's raw bytes from its (untouched) source file, and fsync
    it so its data is durable before the commit marker is created."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for t in tensors:
        header[t["name"]] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, offset + t["nbytes"]],
        }
        offset += t["nbytes"]

    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad the header with spaces so the data section is 8-byte aligned. The spec
    # permits trailing 0x20 padding; alignment keeps every reader happy.
    hbytes += b" " * ((-len(hbytes)) % 8)

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hbytes)))
        out.write(hbytes)
        for t in tensors:
            with open(os.path.join(src_dir, t["src_file"]), "rb") as src:
                src.seek(t["src_begin"])
                remaining = t["nbytes"]
                while remaining:
                    chunk = src.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(
                            f"{t['src_file']}: unexpected EOF reading tensor "
                            f"'{t['name']}' ({remaining} bytes short)")
                    out.write(chunk)
                    remaining -= len(chunk)
        out.flush()
        os.fsync(out.fileno())


def read_marker(staging):
    """Return the committed final filename set, or None if absent/unreadable."""
    marker = os.path.join(staging, COMMIT_MARKER)
    if not os.path.exists(marker):
        return None
    try:
        with open(marker) as f:
            names = json.load(f).get("final_names")
        return set(names) if names else None
    except (ValueError, OSError):
        return None


def commit_staging(staging, models_dir):
    """Replace the top-level weights with the staged set. Re-entrant: deletion
    decisions come from the immutable marker set, and staged files are moved into
    place BEFORE any original is removed, so finishing an interrupted commit never
    loses data. Only call when the marker is present and valid."""
    final_names = read_marker(staging)
    if not final_names:
        raise ValueError(f"{staging}: missing or invalid commit marker")

    # 1) Move staged files up first (same filesystem -> atomic renames). Files
    #    already moved by a prior interrupted commit are simply absent here.
    for f in os.listdir(staging):
        if f == COMMIT_MARKER:
            continue
        os.replace(os.path.join(staging, f), os.path.join(models_dir, f))

    # 2) Drop originals that are not part of the new set (decided from the
    #    immutable marker, not the now-empty staging dir).
    for e in os.listdir(models_dir):
        if e == STAGING_DIR or e in final_names:
            continue
        full = os.path.join(models_dir, e)
        if e.endswith(".safetensors") and os.path.isfile(full):
            os.remove(full)
    # 3) If the new set has no index (single shard), drop any stale one.
    if INDEX_NAME not in final_names:
        stale_idx = os.path.join(models_dir, INDEX_NAME)
        if os.path.exists(stale_idx):
            os.remove(stale_idx)

    fsync_dir(models_dir)

    # 4) Remove the marker and staging dir last.
    os.remove(os.path.join(staging, COMMIT_MARKER))
    os.rmdir(staging)


def main():
    models_dir = os.environ.get("MODELS_DIR", "/models")
    raw_target = os.environ.get("MAX_SHARD_SIZE", "4G")
    target = parse_size(raw_target)
    if not os.path.isdir(models_dir):
        print(f"reshard: {models_dir} is not a directory, nothing to do.")
        return 0

    # Resolve any prior interrupted run FIRST, before the size check -- recovery
    # must happen even when resharding is disabled, or a half-committed model
    # could be left unresolved.
    staging = os.path.join(models_dir, STAGING_DIR)
    if os.path.isdir(staging):
        if read_marker(staging):
            print("reshard: finishing a commit interrupted by a previous run...")
            commit_staging(staging, models_dir)
            print("reshard: recovered; resharded weights are in place.")
            return 0
        # No valid marker -> the write was incomplete; originals are intact.
        print("reshard: discarding incomplete staging from a previous run.")
        shutil.rmtree(staging)

    if target <= 0:
        print("reshard: MAX_SHARD_SIZE disabled, skipping reshard.")
        return 0

    # Top-level safetensors only -- matches the build's per-shard layering, which
    # uses `find -maxdepth 1`. (Subdirectory weights are copied as whole layers.)
    entries = sorted(
        e for e in os.listdir(models_dir)
        if e.endswith(".safetensors")
        and os.path.isfile(os.path.join(models_dir, e))
    )
    if not entries:
        print("reshard: no top-level *.safetensors found, nothing to do.")
        return 0

    sizes = {e: os.path.getsize(os.path.join(models_dir, e)) for e in entries}
    biggest = max(sizes.values())
    if biggest <= target:
        print(f"reshard: all {len(entries)} shard(s) already <= {human(target)} "
              f"(largest {human(biggest)}); nothing to do.")
        return 0

    # Build a global, ordered tensor list (by filename, then on-disk offset, so
    # the output preserves natural model order and reads sequentially).
    merged_metadata = {}
    tensors = []
    file_counts = {}
    for fname in entries:
        header, data_start = read_header(os.path.join(models_dir, fname))
        meta = header.get("__metadata__")
        if isinstance(meta, dict):
            for k, v in meta.items():
                merged_metadata.setdefault(k, v)
        items = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            items.append({
                "name": name,
                "dtype": info["dtype"],
                "shape": info["shape"],
                "src_file": fname,
                "src_begin": data_start + begin,
                "nbytes": end - begin,
            })
        items.sort(key=lambda t: t["src_begin"])
        file_counts[fname] = len(items)
        tensors.extend(items)

    if not tensors:
        print("reshard: headers contain no tensors, nothing to do.")
        return 0

    # A file only benefits from resharding if it is oversized AND holds more than
    # one tensor (a single tensor cannot be split below its own size). If no file
    # is both, we are already as small as possible -> no-op. Keeps the step
    # idempotent even when an individual tensor legitimately exceeds the target.
    if not any(sizes[f] > target and file_counts[f] > 1 for f in entries):
        stuck = [f for f in entries if sizes[f] > target]
        if stuck:
            print(f"reshard: WARNING {len(stuck)} shard(s) exceed {human(target)} "
                  f"but each holds a single tensor that cannot be split further "
                  f"(largest {human(biggest)}); a registry may still reject them.",
                  file=sys.stderr)
        print(f"reshard: no shard can be split below {human(target)}; nothing to do.")
        return 0

    total_size = sum(t["nbytes"] for t in tensors)

    # Disk preflight: the staged copy coexists with the untouched originals until
    # the commit, so we need roughly `total_size` of additional free space.
    needed = total_size + (1 << 30)  # + 1 GiB margin
    st = os.statvfs(models_dir)
    free = st.f_bavail * st.f_frsize
    if free < needed:
        print(f"reshard: ERROR insufficient free space in {models_dir}: need "
              f"~{human(needed)} free (staged copy coexists with the originals), "
              f"have {human(free)}.", file=sys.stderr)
        return 1

    print(f"reshard: target <= {human(target)}/shard; largest current shard is "
          f"{human(biggest)}. Repacking {len(entries)} file(s) "
          f"({human(total_size)})...")

    # Greedily pack tensors into shards no larger than the target. A single tensor
    # bigger than the target cannot be split, so it gets its own (oversized) shard.
    shards = []
    cur, cur_size = [], 0
    for t in tensors:
        nb = t["nbytes"]
        if nb > target:
            if cur:
                shards.append(cur)
                cur, cur_size = [], 0
            shards.append([t])
            print(f"reshard: WARNING tensor '{t['name']}' is {human(nb)} > "
                  f"{human(target)}; it gets its own oversized shard.",
                  file=sys.stderr)
            continue
        if cur and cur_size + nb > target:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(t)
        cur_size += nb
    if cur:
        shards.append(cur)

    n = len(shards)
    if n == 1:
        names = ["model.safetensors"]
    else:
        names = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

    # Write the new shards into staging; originals stay untouched.
    os.mkdir(staging)
    for shard_tensors, final_name in zip(shards, names):
        write_shard(os.path.join(staging, final_name), models_dir,
                    shard_tensors, merged_metadata)
        shard_bytes = sum(t["nbytes"] for t in shard_tensors)
        print(f"  staged {final_name}  ({len(shard_tensors)} tensors, "
              f"{human(shard_bytes)})")

    final_names = list(names)
    if n > 1:
        weight_map = {}
        for shard_tensors, final_name in zip(shards, names):
            for t in shard_tensors:
                weight_map[t["name"]] = final_name
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(os.path.join(staging, INDEX_NAME), "w") as f:
            json.dump(index, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        final_names.append(INDEX_NAME)

    # All shard data is fsync'd; make their directory entries durable, then write
    # and fsync the marker (recording the complete final set) so that "marker
    # present and valid" durably implies "staged set is complete".
    fsync_dir(staging)
    with open(os.path.join(staging, COMMIT_MARKER), "w") as f:
        json.dump({"final_names": final_names}, f)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(staging)

    commit_staging(staging, models_dir)

    if n == 1:
        print(f"reshard: done -> 1 shard 'model.safetensors' "
              f"({human(total_size)}); removed weight index.")
    else:
        print(f"reshard: done -> {n} shards <= {human(target)} each "
              f"({human(total_size)} total); wrote {INDEX_NAME}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
