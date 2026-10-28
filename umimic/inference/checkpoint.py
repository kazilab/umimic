"""Resumable emcee runs.

An :class:`MCMCCheckpoint` saves an ensemble run between steps: the phase
(warmup or production), the steps completed in it, the latest walker state
including emcee's random state, and the production draws so far. Resuming
from it continues the same random stream, so an interrupted run that resumes
produces exactly the draws an uninterrupted run would, given the same
inputs, code and environment.

A checkpoint is only reused when its identity matches the run asking for it:
the caller's identity (for example data and configuration hashes) plus the
sampler's own settings, seed and the emcee and umimic versions. Anything else
starts a fresh run and is reported, never silently merged; with ``strict=True``
it raises :class:`CheckpointMismatch` instead, so a caller that promised to
continue a run never restarts it quietly. The file is a NumPy archive read
without pickle, so a checkpoint cannot execute code when loaded.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1


class CheckpointMismatch(RuntimeError):
    """A strict checkpoint exists but cannot continue this run."""


@dataclass
class SavedRun:
    """What a compatible checkpoint restores."""

    phase: str  # "warmup" or "production"
    steps_done: int
    coords: np.ndarray
    log_prob: np.ndarray
    blobs: np.ndarray | None
    random_state: tuple
    # Production draws so far; empty during warmup.
    chain: np.ndarray
    chain_log_prob: np.ndarray
    chain_blobs: np.ndarray | None
    accepted: np.ndarray


class MCMCCheckpoint:
    """Where and how often an emcee run is saved, and who it belongs to.

    Args:
        path: File to write (a ``.npz`` archive). Its directory is created.
        identity: JSON-serialisable values that must match for a resume,
            such as hashes of the data and configuration.
        every: Save after this many steps within a phase, and at the end of
            each phase.
        strict: Raise :class:`CheckpointMismatch` when a checkpoint file exists
            but cannot be used, instead of starting afresh.
    """

    def __init__(
        self,
        path: str | Path,
        identity: dict[str, Any] | None = None,
        every: int = 50,
        *,
        strict: bool = False,
    ):
        if every < 1:
            raise ValueError("every must be at least 1")
        self.path = Path(path)
        self.identity = dict(identity or {})
        self.every = int(every)
        self.strict = strict
        #: Set by :meth:`load` when a file existed but could not be used.
        self.discarded: str | None = None

    def load(self, run_identity: dict[str, Any]) -> SavedRun | None:
        """The saved run if it exists and matches ``run_identity``, else None."""
        self.discarded = None
        if not self.path.is_file():
            return None
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["meta"]))
                if meta.get("format") != FORMAT_VERSION:
                    self.discarded = f"checkpoint format {meta.get('format')!r} is not {FORMAT_VERSION}"
                    return None
                if meta.get("identity") != _jsonable(run_identity):
                    self.discarded = "checkpoint belongs to a different run (inputs, settings or versions differ)"
                    return None
                rng = meta["random_state"]
                return SavedRun(
                    phase=meta["phase"],
                    steps_done=int(meta["steps_done"]),
                    coords=archive["coords"],
                    log_prob=archive["log_prob"],
                    blobs=archive["blobs"] if "blobs" in archive.files else None,
                    random_state=(
                        rng["name"],
                        archive["rng_keys"],
                        int(rng["pos"]),
                        int(rng["has_gauss"]),
                        float(rng["cached_gaussian"]),
                    ),
                    chain=archive["chain"],
                    chain_log_prob=archive["chain_log_prob"],
                    chain_blobs=archive["chain_blobs"] if "chain_blobs" in archive.files else None,
                    accepted=archive["accepted"],
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.discarded = f"checkpoint could not be read ({exc})"
            return None

    def peek(self) -> dict[str, Any] | None:
        """Phase and steps of a saved run, without checking whose it is."""
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["meta"]))
            return {"phase": str(meta["phase"]), "steps_done": int(meta["steps_done"])}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, run_identity: dict[str, Any], phase: str, steps_done: int, state, backend) -> None:
        """Atomically write the run after ``steps_done`` steps of ``phase``."""
        name, keys, pos, has_gauss, cached = state.random_state
        meta = {
            "format": FORMAT_VERSION,
            "identity": _jsonable(run_identity),
            "phase": phase,
            "steps_done": int(steps_done),
            "random_state": {
                "name": str(name),
                "pos": int(pos),
                "has_gauss": int(has_gauss),
                "cached_gaussian": float(cached),
            },
        }
        arrays: dict[str, np.ndarray] = {
            "meta": np.array(json.dumps(meta)),
            "coords": np.asarray(state.coords),
            "log_prob": np.asarray(state.log_prob),
            "rng_keys": np.asarray(keys),
        }
        if state.blobs is not None:
            arrays["blobs"] = np.asarray(state.blobs)
        if phase == "production":
            n = backend.iteration
            arrays["chain"] = backend.chain[:n]
            arrays["chain_log_prob"] = backend.log_prob[:n]
            arrays["accepted"] = backend.accepted
            if backend.blobs is not None:
                arrays["chain_blobs"] = backend.blobs[:n]
        else:
            arrays["chain"] = np.empty((0,) + np.shape(state.coords))
            arrays["chain_log_prob"] = np.empty((0, len(state.log_prob)))
            arrays["accepted"] = np.zeros(len(state.log_prob))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        # On Windows a virus scanner or indexer can hold the previous file
        # open for a moment, and the replace is refused until it lets go.
        for attempt in range(5):
            try:
                os.replace(temporary, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * 2**attempt)


def _jsonable(value: Any) -> Any:
    """Normalise an identity through JSON so saved and current values compare equal."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))
