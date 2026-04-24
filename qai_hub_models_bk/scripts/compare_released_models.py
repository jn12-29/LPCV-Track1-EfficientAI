#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
compare_release_models.py

This script compares the directories under <repo_root>/qai_hub_models/models
between two git tags in the ai-hub-models repository. It identifies new models
introduced in the second tag and models removed from the first tag.

Usage:
    ./compare_release_models.py <tag1> <tag2>

Arguments:
    <tag1>  The first git tag to compare.
    <tag2>  The second git tag to compare.

Example:
    ./compare_release_models.py v1.0.0 v2.0.0

The script will clone the repository to /tmp/ai-hub-models if it doesn't
already exist, checkout the specified tags, and list the new and removed models.
"""

import os
import subprocess
import sys


def get_directories(repo_root: str, tag: str) -> set[str]:
    subprocess.run(["git", "checkout", "main"], check=True)
    subprocess.run(["git", "pull", "origin", "main"], check=True)
    subprocess.run(["git", "fetch", "--tags"], check=True)
    subprocess.run(["git", "checkout", tag], check=True)
    models_path = os.path.join(repo_root, "src", "qai_hub_models", "models")
    if not os.path.exists(models_path):
        models_path = os.path.join(repo_root, "qai_hub_models", "models")
    return set(os.listdir(models_path))


def main(tag1: str, tag2: str) -> None:
    repo_url = "https://github.com/quic/ai-hub-models.git"
    repo_root = "/tmp/ai-hub-models"

    # Check if the repository already exists
    if not os.path.exists(repo_root):
        # Clone the repository to /tmp/
        subprocess.run(["git", "clone", repo_url, repo_root], check=True)
    os.chdir(repo_root)

    # Get directories for tag1
    dirs_tag1 = get_directories(repo_root, tag1)

    # Get directories for tag2
    dirs_tag2 = get_directories(repo_root, tag2)

    # Find new and removed models
    new_models = dirs_tag2 - dirs_tag1
    removed_models = dirs_tag1 - dirs_tag2

    print(f"New models in {tag2}:")
    for model in new_models:
        print(f"- {model}")

    print(f"\nRemoved models in {tag2}:")
    for model in removed_models:
        print(f"- {model}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_release_models.py <tag1> <tag2>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
