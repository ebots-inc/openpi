#!/usr/bin/env python3
import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download

# ============================================================
# CONFIG: edit these per use (same style as your current files)
# ============================================================

CONFIG = {
    "upload_dataset": {
        "repo_id": "EbotsVLA/pickUp_jointStates_cartStates_stage_v2_v3_v4_merged_15fps_97pct_224res",
        "repo_type": "dataset",
        "local_folder": "/root/.cache/huggingface/lerobot/EbotsVLA/fixed_datasets/pickUp_jointStates_cartStates_stage_v2_v3_v4_merged_15fps_224res",
        "path_in_repo": "./",
        "commit_message": "Uploaded a new dataset: pickUp_jointStates_cartStates_stage_v2_v3_v4_merged_15fps_97pct_224res",
        "private": True,
    },
    "download_dataset": {
        "repo_id": "EbotsVLA/pickUp_jointStates_cartStates_stage_v4Config_0pct_224res",
        "repo_type": "dataset",
        "local_dir": "./.cache/huggingface/lerobot/EbotsVLA/pickUp_jointStates_cartStates_stage_v4Config_0pct_224res",
        "allow_patterns": ["**"],
        "revision": None,  # optional: "main" or a tag/sha
    },
    "upload_model": {
        "repo_id": "EbotsVLA/gr00t_models",
        "repo_type": "model",
        "local_folder": "/home/gayatrid/.cache/huggingface/lerobot/EbotsVLA/checkpoints/gr00t_cart_fv4_pv2v3v4_25k/checkpoint-7500",
        "path_in_repo": "./cart/v2v3v4_25k_ft_v4/checkpoint-7500",
        "commit_message": "Uploaded a new model: gr00t_cart_fv4_7500_pv2v3v4_25k",
        "private": False,
    },
    "download_model": {
        "repo_id": "EbotsVLA/ebots_checkpoints05_cart_v3_97pct_single_thres",
        "repo_type": "model",
        "local_dir": "/home/gayatrid/checkpoints/pi05_ebots_cart/ebots_checkpoints05_cart_v3_97pct_single_thres",
        "allow_patterns": ["**"],
        "revision": None,  # optional
    },
    "tag_dataset": {
        "repo_id": "EbotsVLA/pickUp_jointStates_cartStates_stage_axisAngle_v2_v3_merged",
        "tag": "v2.1",
        "repo_type": "dataset",
    },
}

# ============================================================
# DEFAULTS
# ============================================================
os.environ.setdefault("HF_HUB_UPLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def require_token():
    if not os.getenv("HUGGINGFACE_HUB_TOKEN") and not os.getenv("HF_TOKEN"):
        raise SystemExit("ERROR: Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) before uploading/tagging.")


def upload(cfg):
    require_token()
    create_repo(
        repo_id=cfg["repo_id"],
        repo_type=cfg["repo_type"],
        private=cfg["private"],
        exist_ok=True,
    )
    upload_folder(
        folder_path=cfg["local_folder"],
        repo_id=cfg["repo_id"],
        repo_type=cfg["repo_type"],
        path_in_repo=cfg.get("path_in_repo", "./"),
        commit_message=cfg["commit_message"],
    )
    print(f"✅ Upload complete: {cfg['repo_type']} {cfg['repo_id']}")


def download(cfg):
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type=cfg["repo_type"],
        allow_patterns=cfg.get("allow_patterns", ["**"]),
        local_dir=cfg["local_dir"],
        revision=cfg.get("revision"),
    )
    print(f"✅ Download complete: {cfg['repo_type']} {cfg['repo_id']} -> {cfg['local_dir']}")


def tag_dataset(cfg):
    require_token()
    api = HfApi()
    api.create_tag(
        repo_id=cfg["repo_id"],
        tag=cfg["tag"],
        repo_type="dataset",
    )
    print(f"✅ Tag created: dataset {cfg['repo_id']} -> {cfg['tag']}")


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python hf_run.py <action>\n"
            "Actions:\n"
            "  upload_dataset | download_dataset | upload_model | download_model | tag_dataset\n"
        )
        sys.exit(2)

    action = sys.argv[1]
    if action not in CONFIG:
        raise SystemExit(f"ERROR: Unknown action '{action}'. Valid: {', '.join(CONFIG.keys())}")

    cfg = CONFIG[action]

    if action.startswith("upload_"):
        upload(cfg)
    elif action.startswith("download_"):
        download(cfg)
    elif action == "tag_dataset":
        tag_dataset(cfg)
    else:
        raise SystemExit(f"ERROR: Unhandled action '{action}'")


if __name__ == "__main__":
    main()
