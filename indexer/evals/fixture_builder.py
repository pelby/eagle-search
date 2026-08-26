"""Build the owner-local, sealed fixture manifest for caption selection.

This module never opens an image and never writes a manifest.  The caller is
responsible for placing its returned private payload below ``~/.eagle-search``.
Only ``private_manifest_summary`` is safe to retain outside that private snapshot.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .runner import seal_hidden_manifest


_HASH_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_CANDIDATE_KEYS = {"image_hash", "image_path", "stratum"}
_REQUIRED_KEYS = {"required_stage", "required_role"}
_STAGE_SHAPES = {
    ("A", "target"): 6,
    ("B", "target"): 24,
    ("B", "distractor"): 72,
    ("C", "target"): 120,
    ("C", "distractor"): 120,
}


def _require_candidate(record: Mapping[str, Any]) -> dict[str, str]:
    keys = set(record)
    if keys not in (_CANDIDATE_KEYS, _CANDIDATE_KEYS | _REQUIRED_KEYS):
        raise ValueError(
            "fixture candidates may contain image_hash, image_path and stratum, "
            "plus a complete required_stage/required_role anchor"
        )
    image_hash = record["image_hash"]
    image_path = record["image_path"]
    stratum = record["stratum"]
    if not isinstance(image_hash, str) or not _HASH_RE.fullmatch(image_hash):
        raise ValueError("fixture candidate image_hash must be a sha256 digest")
    if not isinstance(image_path, str) or not image_path.strip() or not isinstance(stratum, str) or not stratum.strip():
        raise ValueError("fixture candidate image_path and stratum must be nonblank strings")
    normalised = {"image_hash": image_hash, "image_path": image_path, "stratum": stratum}
    if _REQUIRED_KEYS.issubset(keys):
        required_stage = record["required_stage"]
        required_role = record["required_role"]
        if (
            not isinstance(required_stage, str)
            or not isinstance(required_role, str)
            or required_stage not in {"B", "C"}
            or required_role != "target"
        ):
            raise ValueError("required candidates must be B or C targets")
        normalised.update({"required_stage": required_stage, "required_role": required_role})
    return normalised


def _balanced_counts(count: int, strata: Sequence[str]) -> dict[str, int]:
    if count < 0 or not strata:
        raise ValueError("positive stratification inputs are required")
    base, remainder = divmod(count, len(strata))
    return {stratum: base + (index < remainder) for index, stratum in enumerate(strata)}


def _take_stratified(
    pools: Mapping[str, list[dict[str, str]]],
    *,
    count: int,
    rng: random.Random,
    required: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    strata = sorted(pools)
    wanted = _balanced_counts(count, strata)
    required_counts = Counter(candidate["stratum"] for candidate in required)
    for stratum, required_count in required_counts.items():
        if stratum not in wanted or required_count > wanted[stratum]:
            raise ValueError(f"required target candidates exceed the balanced quota for stratum {stratum!r}")
    selected = list(required)
    for stratum in strata:
        pool = pools[stratum]
        sample_count = wanted[stratum] - required_counts[stratum]
        if len(pool) < sample_count:
            raise ValueError(f"insufficient candidates in stratum {stratum!r} for a balanced frozen fixture")
        selected.extend(pool.pop() for _ in range(sample_count))
    rng.shuffle(selected)
    return selected


def _seal_payload(
    *, snapshot_id: str, seed: int, fixtures: Sequence[Mapping[str, str]], sealing_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    hidden = [
        {"fixture_id": fixture["fixture_id"], "image_hash": fixture["image_hash"], "stratum": fixture["stratum"], "role": fixture["role"]}
        for fixture in fixtures
        if fixture["stage"] == "C"
    ]
    return {
        "snapshot_id": snapshot_id,
        "seed": seed,
        "hidden_fixtures": hidden,
        "sealing_inputs": dict(sealing_inputs),
    }


def build_private_manifest(
    candidates: Sequence[Mapping[str, Any]],
    *,
    snapshot_id: str,
    seed: int,
    sealing_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically allocate the fixed A/B/C corpus without reading image bytes.

    Candidate order is deliberately discarded; only fixed seed plus hash/stratum
    determine allocation. This prevents an incidental library enumeration order
    from changing a sealed study.
    """

    if not isinstance(snapshot_id, str) or not snapshot_id.strip() or not isinstance(seed, int):
        raise ValueError("snapshot_id and integer frozen seed are required")
    required_seal = {"labels_hash", "gates_hash", "amendment_hash", "instrument_version", "target_count"}
    if set(sealing_inputs) != required_seal:
        raise ValueError("hidden sealing inputs must be frozen labels, gates, amendment, instrument and target count")
    target_count = sealing_inputs["target_count"]
    if not isinstance(target_count, int) or not 60 <= target_count <= 120:
        raise ValueError("hidden target_count must be between 60 and 120")
    normalised = [_require_candidate(candidate) for candidate in candidates]
    if len({candidate["image_hash"] for candidate in normalised}) != len(normalised):
        raise ValueError("fixture candidates must have unique content hashes")
    strata = sorted({candidate["stratum"] for candidate in normalised})
    if len(strata) < 5:
        raise ValueError("private fixture requires at least five visual strata")
    pools: dict[str, list[dict[str, str]]] = {stratum: [] for stratum in strata}
    required: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for candidate in sorted(normalised, key=lambda item: (item["stratum"], item["image_hash"])):
        if "required_stage" in candidate:
            required[(candidate["required_stage"], candidate["required_role"])].append(candidate)
        else:
            pools[candidate["stratum"]].append(candidate)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    fixtures: list[dict[str, str]] = []
    fixture_number = 1
    for (stage, role), count in _STAGE_SHAPES.items():
        for candidate in _take_stratified(
            pools,
            count=count,
            rng=rng,
            required=required.get((stage, role), ()),
        ):
            fixtures.append(
                {
                    "fixture_id": f"f-{fixture_number:04d}",
                    "image_hash": candidate["image_hash"],
                    "image_path": candidate["image_path"],
                    "stratum": candidate["stratum"],
                    "stage": stage,
                    "role": role,
                }
            )
            fixture_number += 1
    manifest: dict[str, Any] = {
        "snapshot_version": 1,
        "snapshot_id": snapshot_id,
        "seed": seed,
        "fixtures": fixtures,
        "hidden_sealing_inputs": dict(sealing_inputs),
    }
    manifest["hidden_seal"] = seal_hidden_manifest(
        _seal_payload(snapshot_id=snapshot_id, seed=seed, fixtures=fixtures, sealing_inputs=sealing_inputs)
    )
    validate_private_manifest(manifest)
    return manifest


def validate_private_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate fixed counts, isolated roles and the seal before a run starts."""

    expected = {"snapshot_version", "snapshot_id", "seed", "fixtures", "hidden_sealing_inputs", "hidden_seal"}
    if set(manifest) != expected or manifest["snapshot_version"] != 1 or not isinstance(manifest["seed"], int):
        raise ValueError("invalid private fixture manifest envelope")
    if not isinstance(manifest["fixtures"], list) or not isinstance(manifest["hidden_sealing_inputs"], Mapping):
        raise ValueError("invalid private fixture manifest payload")
    fixture_keys = {"fixture_id", "image_hash", "image_path", "stratum", "stage", "role"}
    counts: Counter[tuple[str, str]] = Counter()
    identifiers: set[str] = set()
    hashes: set[str] = set()
    for fixture in manifest["fixtures"]:
        if not isinstance(fixture, Mapping) or set(fixture) != fixture_keys:
            raise ValueError("private fixture has malformed fields")
        if fixture["stage"] not in {"A", "B", "C"} or fixture["role"] not in {"target", "distractor"}:
            raise ValueError("private fixture has invalid stage or role")
        if fixture["fixture_id"] in identifiers or fixture["image_hash"] in hashes:
            raise ValueError("private fixture IDs and content hashes must be unique")
        _require_candidate({key: fixture[key] for key in _CANDIDATE_KEYS})
        identifiers.add(fixture["fixture_id"])
        hashes.add(fixture["image_hash"])
        counts[(fixture["stage"], fixture["role"])] += 1
    if counts != Counter(_STAGE_SHAPES):
        raise ValueError("private fixture manifest does not match frozen stage counts and roles")
    expected_seal = seal_hidden_manifest(
        _seal_payload(
            snapshot_id=manifest["snapshot_id"],
            seed=manifest["seed"],
            fixtures=manifest["fixtures"],
            sealing_inputs=manifest["hidden_sealing_inputs"],
        )
    )
    if manifest["hidden_seal"] != expected_seal:
        raise ValueError("private hidden fixture seal does not match frozen inputs")


def private_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return an aggregate suitable for a report without identifiers or private data."""

    validate_private_manifest(manifest)
    counts = Counter((fixture["stage"], fixture["role"]) for fixture in manifest["fixtures"])
    strata = Counter(fixture["stratum"] for fixture in manifest["fixtures"])
    anonymous_strata = {
        f"stratum-{index:02d}": count
        for index, (_name, count) in enumerate(sorted(strata.items()), start=1)
    }
    return {
        "snapshot_version": 1,
        "seed": manifest["seed"],
        "hidden_seal": manifest["hidden_seal"],
        "counts": {f"{stage}:{role}": counts[(stage, role)] for stage, role in sorted(counts)},
        "strata_counts": anonymous_strata,
    }
