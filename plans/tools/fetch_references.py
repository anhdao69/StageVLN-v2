#!/usr/bin/env python3
"""Export selected reference source files without installing/executing project code.

Requires Python >=3.10, Git and network access for first collection. Existing locks
pin Git commits and metadata hashes. --offline only verifies already collected files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(value: str) -> str:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not value or "\\" in value:
        raise ValueError(f"Unsafe export path: {value!r}")
    return p.as_posix()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
        temporary = Path(f.name)
    temporary.replace(path)


def git(*args: str) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=180,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def verify_exports(output: Path, lock: dict, only: set[str]) -> int:
    count = 0
    entries = lock.get("repositories", {}) | lock.get("metadata", {})
    for key, entry in entries.items():
        if only and key not in only:
            continue
        for relative, expected in entry["files"].items():
            path = output / safe_path(key) / safe_path(relative)
            if not path.is_file() or digest(path.read_bytes()) != expected:
                raise ValueError(f"Missing or modified reference: {path}")
            count += 1
    if count == 0:
        raise ValueError("No matching locked reference files were available")
    return count


def collect(manifest: dict, output: Path, only: set[str], offline: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lockpath = output / "references.lock.json"
    lock = json.loads(lockpath.read_text()) if lockpath.exists() else {
        "schema_version": 1, "repositories": {}, "metadata": {}
    }
    known = {r["key"] for r in manifest.get("repositories", []) + manifest.get("metadata", [])}
    if only - known:
        raise ValueError(f"Unknown reference keys: {sorted(only - known)}")
    if offline:
        count = verify_exports(output, lock, only)
        print(f"Verified {count} previously collected source/metadata files")
        return lock
    for repo in manifest.get("repositories", []):
        key = safe_path(repo["key"])
        if only and key not in only:
            continue
        previous = lock["repositories"].get(key)
        if previous and previous["url"] != repo["url"]:
            raise ValueError(f"Repository URL changed for locked key {key}")
        wanted = previous["commit"] if previous else repo["ref"]
        cache = output / ".git-cache" / key
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            git("init", "--bare", str(cache))
        git("--git-dir", str(cache), "fetch", "--depth=1", repo["url"], wanted)
        commit = git("--git-dir", str(cache), "rev-parse", "FETCH_HEAD").decode().strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError(f"Unexpected commit identity: {commit}")
        if previous and commit != previous["commit"]:
            raise ValueError(f"Locked commit changed for {key}")
        root_files = git("--git-dir", str(cache), "ls-tree", "--name-only", commit).decode().splitlines()
        licenses = [p for p in root_files if re.match(r"(?i)^(license|licence|copying|notice)(\.|$)", p)]
        files = list(dict.fromkeys(repo["files"] + licenses))
        hashes = {}
        for relative in files:
            relative = safe_path(relative)
            data = git("--git-dir", str(cache), "show", f"{commit}:{relative}")
            hashes[relative] = digest(data)
            write_atomic(output / key / relative, data)
        lock["repositories"][key] = {
            "url": repo["url"], "requested_ref": repo["ref"], "commit": commit, "files": hashes
        }
    for resource in manifest.get("metadata", []):
        key = safe_path(resource["key"])
        if only and key not in only:
            continue
        previous = lock["metadata"].get(key, {})
        hashes, urls = {}, {}
        for item in resource["files"]:
            relative = safe_path(item["path"])
            expected = previous.get("files", {}).get(relative)
            old_url = previous.get("urls", {}).get(relative)
            if old_url and old_url != item["url"]:
                raise ValueError(f"Metadata URL changed for locked item: {key}/{relative}")
            path = output / key / relative
            if expected and path.is_file():
                data = path.read_bytes()
            else:
                request = Request(item["url"], headers={"User-Agent": "SourceReferenceCollector/1.0"})
                with urlopen(request, timeout=60) as response:
                    data = response.read(10_000_001)
                if len(data) > 10_000_000:
                    raise ValueError(f"Refusing unexpectedly large metadata file: {relative}")
            if expected and digest(data) != expected:
                raise ValueError(f"Locked metadata differs: {key}/{relative}")
            hashes[relative] = digest(data)
            urls[relative] = item["url"]
            write_atomic(path, data)
        lock["metadata"][key] = {"urls": urls, "files": hashes}
    write_atomic(lockpath, (json.dumps(lock, indent=2) + "\n").encode())
    count = verify_exports(output, lock, only)
    print(f"Collected and verified {count} source/metadata files; lock: {lockpath}")
    print("No packages were installed and no repository code was executed.")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("references.json"))
    parser.add_argument("--output", type=Path, default=Path("reference_code"))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported reference manifest schema")
        collect(manifest, args.output.resolve(), set(args.only), args.offline)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Reference collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
