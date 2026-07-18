# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import scireason.vlm_ab.blind as blind_module

from scireason.vlm_ab.blind import (
    ARTIFACT_VERSION,
    BlindReviewError,
    ReviewValidationError,
    build_blind_review_packages,
    deblind_reviews,
    load_review_export,
    validate_review_export,
)


def _inputs(root: Path, count: int = 7):
    image_dir = root / "pages"
    image_dir.mkdir(parents=True)
    benchmark = {}
    base = {}
    tuned = {}
    for index in range(count):
        sample_id = f"sample-{index:02d}"
        image = image_dir / f"paper_page_{index:02d}.png"
        image.write_bytes(b"PNG" + bytes([index]))
        benchmark[sample_id] = {
            "task": f"Inspect evidence for item {index}",
            "images": [f"pages/{image.name}"],
            "metadata": {
                "domain": "science",
                "paper_title": f"Paper {index}",
                "creator_rationale": f"PRIVATE-CREATOR-RATIONALE-{index}",
                "review_metadata": {"expected_winner": "tuned"},
                "expectedErrorModes": ["PRIVATE-EXPECTED-ERROR"],
                "model_id": "PRIVATE-MODEL-IN-BENCHMARK",
            },
        }
        base[(sample_id, "original")] = {
            "status": "success",
            "response": f"Raw response one for {index}",
            "model_id": "org/private-base-model",
        }
        tuned[(sample_id, "original")] = {
            "status": "success",
            "response": f"Raw response two for {index}",
            "model_id": "org/private-tuned-model",
        }
    return benchmark, base, tuned


def _valid_export(public: dict, *, preference: str = "left") -> dict:
    assignment_ids = [row["assignment_id"] for row in public["assignments"]]
    return {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": public["experiment_id"],
        "study_fingerprint": public["study_fingerprint"],
        "reviewer_id": public["reviewer_id"],
        "assignments": assignment_ids,
        "responses": [
            {
                "assignment_id": assignment_id,
                "overall_preference": preference,
                "evidence_preference": "right",
                "visual_preference": "tie",
                "temporal_preference": "skip",
                "left_error_tags": ["missed_visual"],
                "right_error_tags": ["wrong_temporal"],
                "confidence": 4,
                "comments": "checked",
            }
            for assignment_id in assignment_ids
        ],
    }


def test_packages_are_blind_deterministic_balanced_and_copy_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root)
    reviewers = ["reviewer-c", "reviewer-a", "reviewer-b"]

    first = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        tmp_path / "first",
        reviewers,
        reviews_per_item=2,
        seed=1729,
        experiment_id="experiment-public-id",
    )
    second = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        tmp_path / "second",
        list(reversed(reviewers)),
        reviews_per_item=2,
        seed=1729,
        experiment_id="experiment-public-id",
    )

    assert first.paired_item_count == 7
    assert first.total_assignment_count == 14
    assert first.owner_mapping_path.parent == first.output_dir
    owner = json.loads(first.owner_mapping_path.read_text(encoding="utf-8"))
    owner_second = json.loads(second.owner_mapping_path.read_text(encoding="utf-8"))
    assert owner == owner_second

    item_reviewers = Counter(row["assignment_id"] for row in owner["assignments"])
    reviewer_loads = Counter(row["reviewer_id"] for row in owner["assignments"])
    assert set(item_reviewers.values()) == {2}
    assert max(reviewer_loads.values()) - min(reviewer_loads.values()) <= 1
    by_reviewer = defaultdict(list)
    for row in owner["assignments"]:
        by_reviewer[row["reviewer_id"]].append(row)
    for rows in by_reviewer.values():
        tuned_left = sum(row["left_arm"] == "tuned" for row in rows)
        assert abs(tuned_left - (len(rows) - tuned_left)) <= 1

    secrets = (
        "org/private-base-model",
        "org/private-tuned-model",
        "PRIVATE-MODEL-IN-BENCHMARK",
        "PRIVATE-CREATOR-RATIONALE",
        "creator_rationale",
        "expectedErrorModes",
        "PRIVATE-EXPECTED-ERROR",
        "model_id",
        "left_arm",
        "right_arm",
        "sample-00",
    )
    for reviewer_id, package in first.reviewer_packages.items():
        public = json.loads(package.assignment_path.read_text(encoding="utf-8"))
        public_second = json.loads(
            second.reviewer_packages[reviewer_id].assignment_path.read_text(encoding="utf-8")
        )
        assert public == public_second
        package_text = package.assignment_path.read_text(encoding="utf-8")
        package_text += package.html_path.read_text(encoding="utf-8")
        assert all(secret not in package_text for secret in secrets)
        assert "Raw response one" in package_text
        assert "Raw response two" in package_text
        assert "validationErrors" in package_text
        for copied in package.image_paths:
            assert copied.exists()
            assert re.fullmatch(r"img_[0-9a-f]{32}\.png", copied.name)
            assert copied.read_bytes().startswith(b"PNG")
            assert copied.parent == package.package_dir / "images"


def test_missing_paired_outputs_block_blind_packaging(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=4)
    base[("sample-01", "original")]["status"] = "failed"
    tuned[("sample-02", "original")]["response"] = ""
    tuned.pop(("sample-03", "original"))
    tuned[("sample-03", "perturbed")] = {
        "status": "success",
        "response": "wrong condition",
    }

    with pytest.raises(BlindReviewError, match="paired outputs are incomplete"):
        build_blind_review_packages(
            benchmark,
            base,
            tuned,
            dataset_root,
            tmp_path / "packages",
            ["r1", "r2"],
            2,
            9,
        )


@pytest.mark.parametrize(
    "reviewers",
    [
        ["reviewer-a", "REVIEWER-A"],
        ["reviewer-a", "reviewer-\u034fa"],
        ["reviewer-a", "reviewer-\u180ba"],
    ],
)
def test_blind_packaging_rejects_spoofed_reviewer_identities(
    tmp_path: Path, reviewers: list[str]
) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=1)

    with pytest.raises(BlindReviewError, match="reviewer"):
        build_blind_review_packages(
            benchmark,
            base,
            tuned,
            dataset_root,
            tmp_path / "packages",
            reviewers,
            reviews_per_item=1,
            seed=9,
        )


def test_generation_error_is_kept_as_a_reviewable_outcome(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=1)
    base[("sample-00", "original")] = {
        "status": "error",
        "raw_response": None,
        "error": {"type": "OutOfMemoryError", "message": "private runtime detail"},
    }

    result = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        tmp_path / "packages",
        ["reviewer"],
        1,
        9,
    )
    public = json.loads(result.for_reviewer("reviewer").assignment_path.read_text(encoding="utf-8"))
    serialized = json.dumps(public)
    assert "generation_status" in serialized
    assert "OutOfMemoryError" in serialized
    assert "private runtime detail" not in serialized


def test_interrupted_package_build_can_be_retried(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=2)
    output = tmp_path / "packages"
    real_copy = blind_module.shutil.copyfile
    failed = False

    def interrupt_once(source, destination):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated interruption")
        return real_copy(source, destination)

    monkeypatch.setattr(blind_module.shutil, "copyfile", interrupt_once)
    with pytest.raises(OSError, match="simulated interruption"):
        build_blind_review_packages(
            benchmark,
            base,
            tuned,
            dataset_root,
            output,
            ["reviewer-a"],
            reviews_per_item=1,
            seed=1,
        )
    assert not output.exists()

    monkeypatch.setattr(blind_module.shutil, "copyfile", real_copy)
    result = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        output,
        ["reviewer-a"],
        reviews_per_item=1,
        seed=1,
    )
    assert result.output_dir == output.resolve()
    assert result.reviewer_packages["reviewer-a"].assignment_path.is_file()


def test_image_path_traversal_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=1)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    benchmark["sample-00"]["images"] = ["../outside.png"]

    with pytest.raises(BlindReviewError, match="escapes dataset_root"):
        build_blind_review_packages(
            benchmark,
            base,
            tuned,
            dataset_root,
            tmp_path / "packages",
            ["reviewer"],
            1,
            1,
        )


def test_review_exports_are_strict_and_incomplete_forms_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=2)
    result = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        tmp_path / "packages",
        ["reviewer"],
        1,
        4,
    )
    package = result.for_reviewer("reviewer")
    public = json.loads(package.assignment_path.read_text(encoding="utf-8"))
    valid = _valid_export(public)
    expected_ids = [row["assignment_id"] for row in public["assignments"]]
    assert (
        validate_review_export(
            valid,
            expected_experiment_id=result.experiment_id,
            expected_reviewer_id="reviewer",
            expected_assignment_ids=expected_ids,
        )
        == valid
    )

    incomplete = copy.deepcopy(valid)
    incomplete["responses"][0]["temporal_preference"] = ""
    with pytest.raises(ReviewValidationError, match="temporal_preference"):
        validate_review_export(incomplete)

    malformed = copy.deepcopy(valid)
    malformed["unexpected"] = True
    with pytest.raises(ReviewValidationError, match="fields do not match"):
        validate_review_export(malformed)

    wrong_assignment = copy.deepcopy(valid)
    wrong_assignment["responses"][0]["assignment_id"] = "item_unknown"
    with pytest.raises(ReviewValidationError, match="out of order"):
        validate_review_export(wrong_assignment)

    export_path = tmp_path / "review.json"
    export_path.write_text(json.dumps(valid), encoding="utf-8")
    assert load_review_export(export_path) == valid
    export_path.write_text('{"artifact_version": 1, "artifact_version": 1}', encoding="utf-8")
    with pytest.raises(ReviewValidationError, match="duplicate JSON object key"):
        load_review_export(export_path)


def test_deblinding_normalizes_preferences_and_preserves_displayed_side(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    benchmark, base, tuned = _inputs(dataset_root, count=2)
    result = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        dataset_root,
        tmp_path / "packages",
        ["reviewer"],
        1,
        8,
        experiment_id="deblind-exp",
    )
    public = json.loads(result.for_reviewer("reviewer").assignment_path.read_text(encoding="utf-8"))
    exported = _valid_export(public, preference="left")
    rows = deblind_reviews(exported, result.owner_mapping_path, blinding_secret=8)
    owner = json.loads(result.owner_mapping_path.read_text(encoding="utf-8"))
    truth = {(row["reviewer_id"], row["assignment_id"]): row for row in owner["assignments"]}

    assert len(rows) == 2
    for row in rows:
        entry = truth[(row["reviewer_id"], row["assignment_id"])]
        assert row["displayed_preference"] == "left"
        assert row["preference"] == entry["left_arm"]
        assert row["overall_preference"] in {"base", "tuned"}
        assert row["tuned_side"] in {"left", "right"}
        assert row["evidence_preference"] == entry["right_arm"]
        assert row["visual_preference"] == "tie"
        assert row["temporal_preference"] == "skip"
        expected_tuned_tags = (
            ["missed_visual"] if row["tuned_side"] == "left" else ["wrong_temporal"]
        )
        assert row["tuned_error_tags"] == expected_tuned_tags
        assert row["sample_id"].startswith("sample-")

    stale = copy.deepcopy(exported)
    stale["study_fingerprint"] = "0" * 64
    with pytest.raises(ReviewValidationError, match="study_fingerprint"):
        deblind_reviews(stale, result.owner_mapping_path, blinding_secret=8)

    tampered_owner = json.loads(result.owner_mapping_path.read_text(encoding="utf-8"))
    first_entry = tampered_owner["assignments"][0]
    first_entry["left_arm"], first_entry["right_arm"] = (
        first_entry["right_arm"],
        first_entry["left_arm"],
    )
    result.owner_mapping_path.write_text(json.dumps(tampered_owner), encoding="utf-8")
    with pytest.raises(ReviewValidationError, match="integrity HMAC"):
        deblind_reviews(exported, result.owner_mapping_path, blinding_secret=8)
