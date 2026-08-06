#!/usr/bin/env python3
"""
Compare same-name files between two folders and check whether contents are exactly identical.

Default behavior:
- Match files by relative path under each root directory.
- Compare file bytes (exact equality).
- Print summary and mismatched file list.

Edit CONFIG and run:
  python3 compare_same_name_files.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


CONFIG = {
    "folder_a": "/home/syl/GICI-IM/results/debug",
    "folder_b": "/home/syl/GICI-IM/results/debug_raw",
    # If True, include hidden files and directories.
    "include_hidden": False,
}


def should_skip(path: Path, include_hidden: bool) -> bool:
    if include_hidden:
        return False
    return any(part.startswith(".") for part in path.parts)


def collect_files(root: Path, include_hidden: bool) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if should_skip(rel, include_hidden):
            continue
        files[str(rel)] = p
    return files


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compare_folders(folder_a: Path, folder_b: Path, include_hidden: bool) -> Tuple[List[str], List[str], List[str], List[str]]:
    files_a = collect_files(folder_a, include_hidden)
    files_b = collect_files(folder_b, include_hidden)

    keys_a = set(files_a.keys())
    keys_b = set(files_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)

    same_name = sorted(keys_a & keys_b)
    same_content: List[str] = []
    different_content: List[str] = []

    for rel in same_name:
        a = files_a[rel]
        b = files_b[rel]

        # Quick check by size first.
        if a.stat().st_size != b.stat().st_size:
            different_content.append(rel)
            continue

        if file_sha256(a) == file_sha256(b):
            same_content.append(rel)
        else:
            different_content.append(rel)

    return same_content, different_content, only_in_a, only_in_b


def main() -> None:
    folder_a = Path(CONFIG["folder_a"]).expanduser()
    folder_b = Path(CONFIG["folder_b"]).expanduser()
    include_hidden = bool(CONFIG.get("include_hidden", False))

    if not folder_a.exists() or not folder_a.is_dir():
        raise FileNotFoundError(f"folder_a not found or not a directory: {folder_a}")
    if not folder_b.exists() or not folder_b.is_dir():
        raise FileNotFoundError(f"folder_b not found or not a directory: {folder_b}")

    same_content, different_content, only_in_a, only_in_b = compare_folders(
        folder_a, folder_b, include_hidden
    )

    print("=== Compare Result ===")
    print(f"folder_a: {folder_a}")
    print(f"folder_b: {folder_b}")
    print(f"same name & same content: {len(same_content)}")
    print(f"same name but different content: {len(different_content)}")
    print(f"only in folder_a: {len(only_in_a)}")
    print(f"only in folder_b: {len(only_in_b)}")

    if different_content:
        print("\n[Different Content Files]")
        for rel in different_content:
            print(rel)

    if only_in_a:
        print("\n[Only In folder_a]")
        for rel in only_in_a:
            print(rel)

    if only_in_b:
        print("\n[Only In folder_b]")
        for rel in only_in_b:
            print(rel)

    if not different_content and not only_in_a and not only_in_b:
        print("\nPASS: All same-name files are exactly identical.")


if __name__ == "__main__":
    main()
