from types import SimpleNamespace

from app.agent.skills import ground_skill_output, skill_registry
from app.agent.verifier import (
    local_material_verification,
    verify_material_candidate,
    verify_program_research_release,
)


def _call(tool_name, result, status="completed"):
    return SimpleNamespace(tool_name=tool_name, result=result, status=status)


def test_program_release_requires_search_identity_and_official_evidence_binding():
    goal = {
        "entities": {
            "university": "Rice University",
            "program": "",
            "degree_level": "master",
            "fields": ["computer science"],
            "countries": [],
        }
    }
    candidate = {
        "id": "rice-cs",
        "university": "Rice University",
        "name": "Master of Computer Science",
        "degree": "Master",
        "field": "Computer Science",
        "official_url": "https://cs.rice.edu/academics/graduate-programs",
    }
    calls = [
        _call(
            "discover_official_programs",
            {
                "items": [candidate],
                "_evaluation": {
                    "verdict": "accepted",
                    "accepted_program_ids": ["rice-cs"],
                },
            },
        ),
        _call(
            "verify_program_official",
            {
                "program_id": "rice-cs",
                "official_url": candidate["official_url"],
                "status": "verified",
                "evidence": [{"quote": "Submit a statement of purpose."}],
                "_evaluation": {"verdict": "accepted"},
            },
        ),
    ]
    accepted = verify_program_research_release(
        goal, {"programs": [candidate]}, calls
    )
    assert accepted["verdict"] == "accepted"
    assert accepted["accepted_program_ids"] == ["rice-cs"]

    wrong_school = {**candidate, "university": "Stanford University"}
    rejected = verify_program_research_release(
        goal, {"programs": [wrong_school]}, calls
    )
    assert rejected["verdict"] == "rejected"
    assert "wrong_university" in rejected["rejected_programs"][0]["issues"]

    unbound_url = {**candidate, "official_url": "https://example.com/program"}
    rejected_url = verify_program_research_release(
        goal, {"programs": [unbound_url]}, calls
    )
    assert "official_url_not_bound_to_evidence" in (
        rejected_url["rejected_programs"][0]["issues"]
    )


def test_program_release_accepts_canonical_school_aliases():
    goal = {
        "entities": {
            "university": "National University of Singapore (NUS)",
            "program": "",
            "degree_level": "master",
            "fields": ["computer science"],
            "countries": [],
        },
        "parameters": {"max_qs_rank": 100, "excluded_program_ids": []},
    }
    candidate = {
        "id": "nus-mcs",
        "university": "National University of Singapore",
        "name": "Master of Computing in Computer Science",
        "degree": "master",
        "field": "Computer Science",
        "country": "Singapore",
        "qs_rank": 8,
        "official_url": "https://www.comp.nus.edu.sg/programmes/pg/mcs/application",
    }
    calls = [
        _call("discover_official_programs", {
            "items": [candidate],
            "_evaluation": {
                "verdict": "accepted",
                "accepted_program_ids": ["nus-mcs"],
            },
        }),
        _call("verify_program_official", {
            "program_id": "nus-mcs",
            "official_url": candidate["official_url"],
            "status": "verified",
            "evidence": [{"quote": "Master of Computing in Computer Science"}],
            "_evaluation": {"verdict": "accepted"},
        }),
    ]
    accepted = verify_program_research_release(
        goal, {"programs": [candidate]}, calls
    )
    assert accepted["verdict"] == "accepted"
    assert accepted["accepted_program_ids"] == ["nus-mcs"]


def test_program_grounding_preserves_degree_and_country_for_independent_verifier():
    skill = skill_registry.get("program-research")
    grounded = ground_skill_output(
        skill,
        {
            "programs": [{"id": "rice-cs"}],
            "sources": [],
            "unverified": [],
            "summary": "candidate",
        },
        {
            "rice-cs": {
                "id": "rice-cs",
                "university": "Rice University",
                "name": "Master of Computer Science",
                "field": "Computer Science",
                "degree": "Master",
                "country": "United States",
                "official_url": "https://cs.rice.edu/academics/graduate-programs",
            }
        },
        [],
    )
    assert grounded["programs"][0]["degree"] == "Master"
    assert grounded["programs"][0]["country"] == "United States"


def test_material_verifier_rejects_sources_outside_context_manifest():
    payload = {
        "context_manifest": {
            "experience_refs": [{"id": "allowed-experience"}],
            "official_evidence_refs": [{"id": "evidence-1"}],
        }
    }
    candidate = {
        "content": "I completed a grounded retrieval project.",
        "source_experience_ids": ["invented-experience"],
    }
    deterministic = local_material_verification(payload, candidate)
    assert deterministic["verdict"] == "rejected"
    assert "unsupported_experience_reference" in deterministic["issues"]

    report = verify_material_candidate(
        payload,
        candidate,
        {
            "verdict": "accepted",
            "summary": "semantic pass",
            "requirement_coverage": [],
            "unsupported_claims": [],
            "conflicts": [],
            "issues": [],
            "confidence": 0.9,
        },
    )
    assert report["verdict"] == "rejected"
    assert "unsupported_experience_reference" in report["issues"]


def test_material_verifier_rejects_semantic_unsupported_claims():
    payload = {
        "context_manifest": {
            "experience_refs": [{"id": "experience-1"}],
            "official_evidence_refs": [{"id": "evidence-1"}],
        }
    }
    candidate = {
        "content": "I improved accuracy by 90%.",
        "source_experience_ids": ["experience-1"],
    }
    report = verify_material_candidate(
        payload,
        candidate,
        {
            "verdict": "rejected",
            "summary": "Unsupported metric.",
            "requirement_coverage": ["goals"],
            "unsupported_claims": ["improved accuracy by 90%"],
            "conflicts": [],
            "issues": ["unsupported_metric"],
            "confidence": 0.98,
        },
    )
    assert report["verdict"] == "rejected"
    assert {"unsupported_metric", "unsupported_claims"} <= set(report["issues"])
