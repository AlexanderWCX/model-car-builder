#!/usr/bin/env python3
"""Re-shard top-level *.safetensors files so no single shard exceeds a target size.

Why: the ModelCar build gives each top-level safetensors file its own OCI layer
(for parallel/resumable pulls). When an upstream repo packs most weights into one
giant file (e.g. a single ~50 GB shard), that becomes one ~50 GB layer -- which
blows past container-registry per-layer limits (Quay's MAXIMUM_LAYER_SIZE
defaults to 20G) and also defeats parallel pulls.

This rewrites the weights into evenly sized shards by repacking at the raw-byte
level -- no torch/numpy, fully dtype-agnostic (bf16/fp8/etc. are copied verbatim)
-- and regenerates the weight index(es) so the build produces many small,
registry-friendly layers.

Multiple weight sets: a repo may ship more than one independent set of weights,
each with its OWN index -- e.g. Mistral models carry both the HF format
(model-0000X-of-Y.safetensors + model.safetensors.index.json) and the Mistral
"consolidated" format (consolidated-0000X-of-Y.safetensors +
consolidated.safetensors.index.json), with DIFFERENT tensor naming. Each index
file plus the shards it references forms a group; groups are resharded
independently (never merged) and each index is regenerated against its own new
shards. Loose safetensors not referenced by any index form their own group.

Properties:
  * Idempotent: a group whose shards are already <= the target is left untouched.
  * Crash-safe: new shards are written into a staging subdirectory while the
    originals are left untouched; a COMMITTED marker recording the complete final
    filename set AND the exact originals to remove is fsync'd into place only
    after all shard data is fsync'd. The commit MOVES staged files into place
    before deleting any original, and deletes only the recorded superseded files
    -- so it is re-entrant (an interrupted commit is finished, not corrupted) and
    never touches a skipped group. This matters because `build.sh restore` MOVES
    weights out of the archive, so models/ can be the only copy.

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
INDEX_SUFFIX = ".safetensors.index.json"
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
    """Return the marker dict {final_names, remove} or None if absent/unreadable."""
    marker = os.path.join(staging, COMMIT_MARKER)
    if not os.path.exists(marker):
        return None
    try:
        with open(marker) as f:
            m = json.load(f)
        if isinstance(m, dict) and m.get("final_names") is not None:
            m.setdefault("remove", [])
            return m
    except (ValueError, OSError):
        pass
    return None


def commit_staging(staging, models_dir, marker):
    """Atomically replace weights with the staged set. Re-entrant: moves staged
    files up first, then deletes only the recorded superseded originals (never a
    file in the new set, never a skipped group's file). Only call with a valid
    marker."""
    final_names = set(marker["final_names"])
    remove = set(marker.get("remove", []))

    # 1) Move staged files up first (already-moved ones from a prior interrupted
    #    commit are simply absent here).
    for f in os.listdir(staging):
        if f == COMMIT_MARKER:
            continue
        os.replace(os.path.join(staging, f), os.path.join(models_dir, f))

    # 2) Delete superseded originals (immutable list from the marker); never a
    #    file that is part of the new set.
    for f in remove:
        if f in final_names:
            continue
        p = os.path.join(models_dir, f)
        if os.path.isfile(p):
            os.remove(p)

    fsync_dir(models_dir)

    # 3) Remove the marker and staging dir last.
    os.remove(os.path.join(staging, COMMIT_MARKER))
    os.rmdir(staging)


def discover_groups(models_dir, safetensors):
    """Group safetensors by the index that references them. Each group:
    {prefix, index_name (or None), files: [shard names]}. Loose safetensors (not
    referenced by any index) form a final group."""
    groups = []
    claimed = set()
    index_files = sorted(
        f for f in os.listdir(models_dir)
        if f.endswith(INDEX_SUFFIX) and os.path.isfile(os.path.join(models_dir, f))
    )
    have = set(safetensors)
    for idxf in index_files:
        try:
            with open(os.path.join(models_dir, idxf)) as f:
                wm = json.load(f).get("weight_map", {})
        except (ValueError, OSError):
            continue
        files = sorted(set(wm.values()) & have)
        if not files:
            continue
        prefix = idxf[:-len(INDEX_SUFFIX)]
        groups.append({"prefix": prefix, "index_name": idxf, "files": files})
        claimed.update(files)

    loose = sorted(f for f in safetensors if f not in claimed)
    if loose:
        taken = {g["prefix"] for g in groups}
        prefix = "model" if "model" not in taken else "weights"
        groups.append({"prefix": prefix, "index_name": None, "files": loose})
    return groups


def build_group_tensors(models_dir, files):
    """Return (tensors, merged_metadata, file_counts) for a group's shard files."""
    merged_metadata = {}
    tensors = []
    file_counts = {}
    for fname in files:
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
    return tensors, merged_metadata, file_counts


def pack(tensors, target, prefix):
    """Greedily pack tensors into shards <= target. A tensor bigger than target
    gets its own (oversized) shard with a warning."""
    shards, cur, cur_size = [], [], 0
    for t in tensors:
        nb = t["nbytes"]
        if nb > target:
            if cur:
                shards.append(cur)
                cur, cur_size = [], 0
            shards.append([t])
            print(f"reshard: WARNING [{prefix}] tensor '{t['name']}' is {human(nb)} "
                  f"> {human(target)}; it gets its own oversized shard.",
                  file=sys.stderr)
            continue
        if cur and cur_size + nb > target:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(t)
        cur_size += nb
    if cur:
        shards.append(cur)
    return shards


def main():
    models_dir = os.environ.get("MODELS_DIR", "/models")
    target = parse_size(os.environ.get("MAX_SHARD_SIZE", "4G"))
    if not os.path.isdir(models_dir):
        print(f"reshard: {models_dir} is not a directory, nothing to do.")
        return 0

    # Resolve any prior interrupted run FIRST -- recovery must happen even when
    # resharding is disabled, or a half-committed model could be left unresolved.
    staging = os.path.join(models_dir, STAGING_DIR)
    if os.path.isdir(staging):
        marker = read_marker(staging)
        if marker:
            print("reshard: finishing a commit interrupted by a previous run...")
            commit_staging(staging, models_dir, marker)
            print("reshard: recovered; resharded weights are in place.")
            return 0
        print("reshard: discarding incomplete staging from a previous run.")
        shutil.rmtree(staging)

    if target <= 0:
        print("reshard: MAX_SHARD_SIZE disabled, skipping reshard.")
        return 0

    safetensors = sorted(
        f for f in os.listdir(models_dir)
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(models_dir, f))
    )
    if not safetensors:
        print("reshard: no top-level *.safetensors found, nothing to do.")
        return 0

    groups = discover_groups(models_dir, safetensors)

    # Plan each group independently. A group is resharded only if it has a file
    # over the target that holds more than one tensor (a single tensor can't be
    # split below its own size). Skipped groups are left exactly as-is.
    plans = []
    for g in groups:
        sizes = {f: os.path.getsize(os.path.join(models_dir, f)) for f in g["files"]}
        biggest = max(sizes.values())
        if biggest <= target:
            continue
        tensors, metadata, file_counts = build_group_tensors(models_dir, g["files"])
        if not tensors:
            continue
        if not any(sizes[f] > target and file_counts[f] > 1 for f in g["files"]):
            print(f"reshard: WARNING [{g['prefix']}] {sum(1 for f in g['files'] if sizes[f] > target)} "
                  f"shard(s) exceed {human(target)} but each holds a single "
                  f"unsplittable tensor; a registry may still reject them.",
                  file=sys.stderr)
            continue
        plans.append({"group": g, "tensors": tensors, "metadata": metadata})

    if not plans:
        print(f"reshard: nothing to do; every weight set is within {human(target)} "
              f"(or cannot be split further).")
        return 0

    total_size = sum(t["nbytes"] for p in plans for t in p["tensors"])
    needed = total_size + (1 << 30)  # staged copy coexists with originals + 1 GiB
    st = os.statvfs(models_dir)
    free = st.f_bavail * st.f_frsize
    if free < needed:
        print(f"reshard: ERROR insufficient free space in {models_dir}: need "
              f"~{human(needed)} free (staged copy coexists with the originals), "
              f"have {human(free)}.", file=sys.stderr)
        return 1

    set_word = "set" if len(plans) == 1 else "sets"
    print(f"reshard: target <= {human(target)}/shard; repacking {len(plans)} weight "
          f"{set_word} ({human(total_size)} total)...")

    os.mkdir(staging)
    final_names = []
    remove = set()
    summaries = []
    for p in plans:
        g = p["group"]
        prefix = g["prefix"]
        shards = pack(p["tensors"], target, prefix)
        n = len(shards)
        if n == 1:
            names = [f"{prefix}.safetensors"]
        else:
            names = [f"{prefix}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

        for shard_tensors, final_name in zip(shards, names):
            write_shard(os.path.join(staging, final_name), models_dir,
                        shard_tensors, p["metadata"])
        final_names.extend(names)

        group_bytes = sum(t["nbytes"] for t in p["tensors"])
        index_name = prefix + INDEX_SUFFIX
        if n > 1:
            weight_map = {}
            for shard_tensors, final_name in zip(shards, names):
                for t in shard_tensors:
                    weight_map[t["name"]] = final_name
            index = {
                "metadata": {"total_size": group_bytes},
                "weight_map": dict(sorted(weight_map.items())),
            }
            with open(os.path.join(staging, index_name), "w") as f:
                json.dump(index, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            final_names.append(index_name)

        # Originals this group supersedes: its old shard files, plus its old index
        # (regenerated when n>1 -> overwritten by the move; dropped when n==1).
        remove.update(g["files"])
        if g["index_name"]:
            remove.add(g["index_name"])
        summaries.append((prefix, n, group_bytes))

    remove = sorted(remove - set(final_names))

    # All shard data is fsync'd; make staging's dir entries durable, then write
    # and fsync the marker so "marker present and valid" durably implies the
    # staged set is complete.
    fsync_dir(staging)
    with open(os.path.join(staging, COMMIT_MARKER), "w") as f:
        json.dump({"final_names": final_names, "remove": remove}, f)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(staging)

    commit_staging(staging, models_dir, {"final_names": final_names, "remove": remove})

    for prefix, n, group_bytes in summaries:
        if n == 1:
            print(f"reshard: [{prefix}] -> 1 shard '{prefix}.safetensors' "
                  f"({human(group_bytes)}); no index needed.")
        else:
            print(f"reshard: [{prefix}] -> {n} shards <= {human(target)} each "
                  f"({human(group_bytes)}); wrote {prefix}{INDEX_SUFFIX}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
