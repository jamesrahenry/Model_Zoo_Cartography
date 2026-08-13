"""
corpus_io.py — Corpus storage with HF as the system of record.

The dataset `james-ra-henry/MZC-Corpus` (private) is canonical for net
weights; the local `corpus/` tree is a working cache. Provenance JSONs stay
local (and in the dataset) — they are the run manifest.

  net_paths(run_id)         local .npz paths for a run, downloading any that
                            were pruned locally (by matching net_*.json)
  upload_run(run_id, prune) upload the run folder, verify every local file
                            appears in the repo listing, then optionally
                            delete local .npz (JSONs are always kept)

Analysis scripts should use net_paths() instead of globbing corpus/ directly,
so they keep working after a prune.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
HF_REPO = "james-ra-henry/MZC-Corpus"


def net_paths(run_id: str) -> list[Path]:
    """Local .npz paths for a run; fetch pruned ones from HF on demand."""
    run_dir = CORPUS_DIR / run_id
    jsons = sorted(run_dir.glob("net_*.json"))
    if not jsons:
        raise FileNotFoundError(f"no provenance JSONs in {run_dir}")
    paths = []
    missing = [j.with_suffix(".npz") for j in jsons if not j.with_suffix(".npz").exists()]
    if missing:
        from huggingface_hub import hf_hub_download
        print(f"corpus_io: fetching {len(missing)} pruned net(s) for {run_id} from {HF_REPO}")
        for npz in missing:
            # local_dir download: file lands directly under corpus/ with no
            # duplicate copy in the HF cache (matters at full-corpus scale)
            hf_hub_download(HF_REPO, f"corpus/{run_id}/{npz.name}",
                            repo_type="dataset",
                            local_dir=str(CORPUS_DIR.parent))
    for j in jsons:
        paths.append(j.with_suffix(".npz"))
    return paths


def upload_run(run_id: str, prune: bool = False) -> None:
    """Upload corpus/<run_id>/ to the dataset, verify, optionally prune .npz."""
    from huggingface_hub import HfApi

    run_dir = CORPUS_DIR / run_id
    local = sorted(p.name for p in run_dir.iterdir() if p.suffix in (".npz", ".json"))
    if not local:
        raise FileNotFoundError(f"nothing to upload in {run_dir}")
    api = HfApi()
    api.upload_folder(repo_id=HF_REPO, repo_type="dataset",
                      folder_path=str(run_dir), path_in_repo=f"corpus/{run_id}")
    remote = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    missing = [f for f in local if f"corpus/{run_id}/{f}" not in remote]
    if missing:
        raise RuntimeError(f"upload verification FAILED for {run_id}: {missing} "
                           "not in repo listing — local files kept")
    print(f"corpus_io: {run_id} uploaded and verified ({len(local)} files)")
    if prune:
        for p in run_dir.glob("net_*.npz"):
            p.unlink()
        print(f"corpus_io: pruned local .npz for {run_id} (JSONs kept)")
