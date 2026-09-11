"""Freeze every non-overlapping Hy test window, excluding table_014."""

import argparse

from evaluation.stage_a.contract import (
    SELECTION_SCHEMA, canonical_sha256, read_identity_contract, write_selection,
)
from evaluation.stage_a.data import HY_EXCLUDED_TABLES, _read_hy_manifest_matches


def all_hy_windows(identity, matched):
    records = []
    included = [r for r in identity["records"] if r["split"] == "test" and r["table_name"] not in HY_EXCLUDED_TABLES]
    for item in sorted(included, key=lambda r: (r["table_name"], r["episode_index"])):
        source = matched[(item["table_name"], item["episode_index"])]
        if int(source["window_count"]) < 1:
            raise ValueError(f"test episode has no valid windows: {item['episode_id']}")
        for window in range(int(source["window_count"])):
            indices = [round((window * 4 + t) * float(source["fps"]) / 10) for t in range(4)]
            if indices[-1] >= int(source["length"]):
                raise ValueError(f"manifest window exceeds episode: {item['episode_id']}")
            records.append({
                "selection_index": len(records), "legacy_episode_id": item["episode_id"],
                "legacy_group": item["table_name"], "canonical_episode_index": item["episode_index"],
                "canonical_rgb_target_length": source["length"], "source_fps": source["fps"],
                "hy_manifest_record": source, "legacy_window_index": window,
                "anchor_rgb_index": window * 4, "expected_frame_offsets": [0, 1, 2, 3],
                "expected_source_frame_indices": indices,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-contract", required=True)
    parser.add_argument("--hy-manifest", required=True)
    parser.add_argument("--hy-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    identity = read_identity_contract(args.identity_contract, dataset_id="hy")
    test = [r for r in identity["records"] if r["split"] == "test"]
    excluded = [r for r in test if r["table_name"] in HY_EXCLUDED_TABLES]
    included = [r for r in test if r["table_name"] not in HY_EXCLUDED_TABLES]
    keys = {(r["table_name"], r["episode_index"]) for r in included}
    matched, manifest = _read_hy_manifest_matches(args.hy_manifest, args.hy_manifest_sha256, keys)
    if set(matched) != keys:
        raise ValueError(f"Hy identity join missing={len(keys - set(matched))}")
    records = all_hy_windows(identity, matched)
    payload = {
        "schema": SELECTION_SCHEMA, "dataset_id": "hy", "split": "test", "seed": 1234,
        "sample_count": len(records), "selection_policy": "all_nonoverlapping_test_windows",
        "semantic_rgb_rate_hz": 10, "data_backend": "hy_lance_manifest", "hy_manifest": manifest,
        "identity_contract": {"path": identity["identity_contract_path"], "sha256": identity["identity_contract_sha256"], "source_manifest_sha256": identity["source_manifest_sha256"]},
        "included_source_groups": sorted({r["table_name"] for r in included}),
        "excluded_source_groups": {"groups": list(HY_EXCLUDED_TABLES), "episode_count": len(excluded), "episode_ids_sha256": canonical_sha256(sorted(r["episode_id"] for r in excluded))},
        "identity_mapping": {"identity_split_episode_count": len(test), "post_exclusion_episode_count": len(included), "mapped_complete_episode_count": len(matched), "missing_episode_count": 0},
        "decode_validation": {"enabled": False, "policy": "decode every selected window during evaluation; fail on any error, never skip"},
        "records": records,
    }
    payload["selection_sha256"] = canonical_sha256(payload)
    write_selection(args.output, payload)
    print(f"Hy: {len(included)} episodes, {len(records)} windows; excluded {len(excluded)} episodes", flush=True)


if __name__ == "__main__":
    main()
