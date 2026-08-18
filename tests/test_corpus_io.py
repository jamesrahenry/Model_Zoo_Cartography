"""Unit tests for the filesystem-only parts of train/corpus_io.py.

net_paths()'s HF-download path and upload_run() both need network + private
HF dataset credentials and are out of scope for unit tests (mock-worthy, but
not attempted here — see the PR description). This file covers the local
logic that doesn't need either.
"""
from __future__ import annotations

import json

import pytest

import corpus_io


class TestNetPaths:
    def test_raises_when_run_dir_has_no_provenance_jsons(self, tmp_path, monkeypatch):
        monkeypatch.setattr(corpus_io, "CORPUS_DIR", tmp_path)
        (tmp_path / "empty_run").mkdir()
        with pytest.raises(FileNotFoundError):
            corpus_io.net_paths("empty_run")

    def test_returns_existing_npz_paths_without_touching_the_network(self, tmp_path, monkeypatch):
        monkeypatch.setattr(corpus_io, "CORPUS_DIR", tmp_path)
        run_dir = tmp_path / "run_a"
        run_dir.mkdir()
        for i in range(3):
            (run_dir / f"net_{i:04d}.json").write_text(json.dumps({"net": i}))
            (run_dir / f"net_{i:04d}.npz").write_bytes(b"")
        paths = corpus_io.net_paths("run_a")
        assert [p.name for p in paths] == [f"net_{i:04d}.npz" for i in range(3)]
        for p in paths:
            assert p.exists()


class TestUploadRun:
    def test_raises_when_run_dir_is_empty(self, tmp_path, monkeypatch):
        # upload_run() imports huggingface_hub before its own emptiness check
        pytest.importorskip("huggingface_hub")
        monkeypatch.setattr(corpus_io, "CORPUS_DIR", tmp_path)
        run_dir = tmp_path / "run_b"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            corpus_io.upload_run("run_b")
