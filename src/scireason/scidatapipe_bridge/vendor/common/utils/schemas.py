"""JSON Schemas for the unified Task 1 trajectory + Task 2 assertion bundle.

Task 1 is frozen at ``artifact_version: 4``; Task 2 uses a flat ``assertions``
list. Both are validated by :func:`scidatapipe.common.utils.io.validate` at
normalization time.
"""
from __future__ import annotations


TASK1_SCHEMA_V4: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SciHist unified task-1 trajectory (v4)",
    "type": "object",
    "required": [
        "artifact_version",
        "topic",
        "domain",
        "cutoff_year",
        "submission_id",
        "papers",
        "steps",
        "edges",
        "expert",
    ],
    "properties": {
        "artifact_version": {"const": 4},
        "topic": {"type": "string"},
        "domain": {"type": "string"},
        "domain_label": {"type": "string"},
        "cutoff_year": {"type": ["integer", "null"]},
        "submission_id": {"type": "string"},
        "artifact_hash": {"type": "string"},
        "generated_at": {"type": "string"},
        "expert": {
            "type": "object",
            "properties": {
                "last_name": {"type": "string"},
                "first_name": {"type": "string"},
                "patronymic": {"type": "string"},
                "full_name": {"type": "string"},
                "latin_full_name": {"type": "string"},
                "latin_slug": {"type": "string"},
            },
        },
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "paper_type"],
                "properties": {
                    "id": {"type": "string"},
                    "paper_type": {"enum": ["arxiv", "doi", "wiki", "url"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "version": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "title": {"type": "string"},
                    "resolved": {"type": "boolean"},
                    "raw": {"type": "string"},
                },
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step_id", "claim", "sources", "inference"],
                "properties": {
                    "step_id": {"type": "integer", "minimum": 1},
                    "claim": {"type": "string"},
                    "importance": {
                        "enum": ["ключевая", "не ключевая", "фоновая"]
                    },
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "time_source": {"type": "string"},
                    "conditions": {
                        "type": "object",
                        "properties": {
                            "system": {"type": "string"},
                            "environment": {"type": "string"},
                            "protocol": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                    },
                    "sources": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["type", "source"],
                            "properties": {
                                "type": {"enum": ["text", "image", "table"]},
                                "source": {"type": "string"},
                                "paper_ref_id": {"type": "string"},
                                "page": {"type": ["integer", "null"]},
                                "locator": {"type": "string"},
                                "snippet_or_summary": {"type": "string"},
                                "has_figure_ref": {"type": "boolean"},
                                "figure_kind": {
                                    "enum": ["figure", "table", ""]
                                },
                                "figure_number": {"type": ["integer", "null"]},
                            },
                        },
                    },
                    "discovery_context": {
                        "type": "object",
                        "properties": {
                            "simultaneous_discovery": {"type": "boolean"},
                            "geography": {
                                "type": "object",
                                "properties": {
                                    "country": {"type": "object"},
                                    "city": {"type": "object"},
                                },
                            },
                            "science_branches": {"type": "array"},
                        },
                    },
                    "inference": {"type": "string"},
                    "next_question": {"type": "string"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from_step_id", "to_step_id"],
                "properties": {
                    "from_step_id": {"type": "integer"},
                    "to_step_id": {"type": "integer"},
                    "predicate": {"type": "string"},
                    "directionality": {
                        "enum": ["directed", "bidirectional", "simultaneous"]
                    },
                    "direction_label": {"type": "string"},
                    "simultaneous_discovery": {"type": "boolean"},
                },
            },
        },
    },
}


TASK2_ASSERTION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SciHist unified task-2 assertion bundle",
    "type": "object",
    "required": ["submission_id", "assertions"],
    "properties": {
        "submission_id": {"type": "string"},
        "trajectory_submission_id": {"type": "string"},
        "domain": {"type": "string"},
        "topic": {"type": "string"},
        "cutoff_year": {"type": ["integer", "null"]},
        "reviewer_id": {"type": "string"},
        "timestamp": {"type": "string"},
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "assertion_id",
                    "graph_kind",
                    "subject",
                    "predicate",
                    "object",
                    "start_date",
                    "end_date",
                ],
                "properties": {
                    "assertion_id": {"type": "string"},
                    "graph_kind": {"enum": ["gold", "auto"]},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "page": {"type": ["integer", "null"]},
                            "figure_or_table": {"type": "string"},
                            "paper_id": {"type": "string"},
                            "image_path": {"type": "string"},
                        },
                    },
                    "paper_ids": {"type": "array"},
                    "importance_score": {"type": ["number", "null"]},
                    "expert": {
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string"},
                            "rationale": {"type": "string"},
                            "corrected_start_date": {"type": "string"},
                            "corrected_end_date": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


__all__ = ["TASK1_SCHEMA_V4", "TASK2_ASSERTION_SCHEMA"]
