"""
Google Drive & Hugging Face streaming dataset storage adapter.
Ensures raw audio corpora live in Google Drive (or stream directly from Hugging Face)
rather than consuming local SSD storage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StorageConfig:
    """
    Resolves dataset storage location with priority:
      1. `VOICE_AGENT_DATA_ROOT` env var (local mounted Google Drive path, e.g. G:/My Drive/VoiceAgent/data)
      2. Auto-detected mounted Google Drive path (G:/My Drive/VoiceAgent/data or G:/我的云端硬盘/VoiceAgent/data)
      3. `VOICE_AGENT_RCLONE_REMOTE` (e.g. `gdrive:VoiceAgent/data`) for headless sync on AutoDL / Windows
    """

    local_mount_root: Path | None
    rclone_remote: str | None


def detect_gdrive_storage() -> StorageConfig:
    env_root = os.getenv("VOICE_AGENT_DATA_ROOT")
    if env_root:
        return StorageConfig(local_mount_root=Path(env_root), rclone_remote=None)

    for candidate in (
        Path("G:/My Drive/VoiceAgent/data"),
        Path("G:/我的云端硬盘/VoiceAgent/data"),
    ):
        if candidate.parent.parent.exists():
            return StorageConfig(local_mount_root=candidate, rclone_remote=None)

    rclone_remote = os.getenv("VOICE_AGENT_RCLONE_REMOTE", "gdrive:VoiceAgent/data")
    return StorageConfig(local_mount_root=None, rclone_remote=rclone_remote)


def sync_artifact_to_gdrive(local_path: str | Path, remote_relpath: str) -> str:
    """Uploads a dataset manifest, feature cache, or checkpoint to Google Drive."""
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"Source artifact not found: {src}")

    cfg = detect_gdrive_storage()
    if cfg.local_mount_root is not None:
        dst = cfg.local_mount_root / remote_relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    if cfg.rclone_remote and shutil.which("rclone"):
        target = f"{cfg.rclone_remote.rstrip('/')}/{remote_relpath.lstrip('/')}"
        subprocess.run(["rclone", "copyto", str(src), target], check=True)
        return target

    raise RuntimeError(
        "Google Drive is not mounted (set VOICE_AGENT_DATA_ROOT=G:/My Drive/VoiceAgent/data) "
        "and `rclone` is not installed/configured."
    )


def open_hf_streaming_dataset(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    token: str | None = None,
) -> Any:
    """Opens a Hugging Face dataset in pure streaming=True mode (zero full local download)."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if token or os.getenv("HF_TOKEN"):
        kwargs["token"] = token or os.getenv("HF_TOKEN")
    if config_name:
        return load_dataset(repo_id, config_name, **kwargs)
    return load_dataset(repo_id, **kwargs)
