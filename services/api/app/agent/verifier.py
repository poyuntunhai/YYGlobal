from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

from app.services.business import canonical_fields
from app.services.rankings import same_university


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").split())


def _degree_matches(actual: Any, expected: str) -> bool:
    if not expected:
        return True
    value = _normalized(actual)
    if expected == "master":
        return value in {"master", "ms", "m s", "mcs", "meng"} or "master" in value
    if expected == "doctoral":
        return value in {"phd", "ph d", "doctoral", "doctorate"} or "doctor" in value
    if expected == "undergraduate":
        return value in {"bachelor", "bs", "b s", "undergraduate"} or "bachelor" in value
    return value == _normalized(expected)


def _identity_issues(goal_spec: Dict[str, Any], program: Dict[str, Any]) -> List[str]:
    entities = goal_spec.get("entities", {})
    issues = []
    university = str(entities.get("university") or "")
    requested_program = _normalized(entities.get("program"))
    degree = str(entities.get("degree_level") or "")
    fields = list(entities.get("fields") or [])
    countries = {_normalized(value) for value in entities.get("countries") or []}
    if university and not same_university(
        str(program.get("university") or ""), university
    ):
        issues.append("wrong_university")
    if requested_program and _normalized(program.get("name")) != requested_program:
        issues.append("wrong_program")
    if degree and not _degree_matches(program.get("degree"), degree):
        issues.append("wrong_degree_level")
    if fields:
        actual_fields = set(canonical_fields([str(program.get("field") or "")]))
        expected_fields = set(canonical_fields([str(value) for value in fields]))
        normalized_actual = _normalized(program.get("field"))
        freeform_match = any(
            _normalized(value) in normalized_actual or normalized_actual in _normalized(value)
            for value in fields
            if _normalized(value)
        )
        if not actual_fields.intersection(expected_fields) and not freeform_match:
            issues.append("wrong_field")
    if countries and _normalized(program.get("country")) not in countries:
        issues.append("wrong_country")
    parameters = goal_spec.get("parameters", {})
    max_qs_rank = parameters.get("max_qs_rank")
    rank = program.get("qs_rank")
    if max_qs_rank and (
        not isinstance(rank, (int, float)) or rank < 1 or rank > max_qs_rank
    ):
        issues.append("outside_qs_range")
    if str(program.get("id") or "") in {
        str(value) for value in parameters.get("excluded_program_ids") or []
    }:
        issues.append("already_displayed")
    return issues


def _hostname(url: Any) -> str:
    return (urlparse(str(url or "")).hostname or "").lower()


def _tool_result_items(call: Any) -> List[Dict[str, Any]]:
    result = call.result or {}
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        if result.get("program_id"):
            return [result]
    return []


def verify_program_research_release(
    goal_spec: Dict[str, Any],
    structured_output: Dict[str, Any],
    tool_calls: Sequence[Any],
) -> Dict[str, Any]:
    """Independently gate final recommendations using persisted tool evidence."""
    searched_ids = set()
    verified_ids = set()
    verified_urls_by_program: Dict[str, set] = {}
    for call in tool_calls:
        result = call.result or {}
        evaluation = result.get("_evaluation", {}) if isinstance(result, dict) else {}
        if call.status != "completed" or evaluation.get("verdict") != "accepted":
            continue
        if call.tool_name in {
            "discover_official_programs", "search_programs", "mcp_catalog_search"
        }:
            searched_ids.update(
                str(value)
                for value in evaluation.get("accepted_program_ids", [])
                if value
            )
        if call.tool_name in {"verify_program_official", "extract_program_requirements"}:
            for item in _tool_result_items(call):
                program_id = str(item.get("program_id") or "")
                if not program_id:
                    continue
                verified_ids.add(program_id)
                urls = {
                    str(item.get("official_url") or ""),
                    *{
                        str(source.get("url") or "")
                        for source in item.get("source_results", [])
                        if isinstance(source, dict)
                    },
                }
                verified_urls_by_program.setdefault(program_id, set()).update(
                    url for url in urls if _hostname(url)
                )

    accepted_programs = []
    rejected_programs = []
    accepted_urls = set()
    for program in structured_output.get("programs") or []:
        program_id = str(program.get("id") or "")
        issues = _identity_issues(goal_spec, program)
        if program_id not in searched_ids:
            issues.append("identity_search_not_verified")
        if program_id not in verified_ids:
            issues.append("official_evidence_not_verified")
        output_host = _hostname(program.get("official_url"))
        evidence_hosts = {
            _hostname(url) for url in verified_urls_by_program.get(program_id, set())
        }
        if not output_host or output_host not in evidence_hosts:
            issues.append("official_url_not_bound_to_evidence")
        if issues:
            rejected_programs.append({"id": program_id or None, "issues": sorted(set(issues))})
        else:
            accepted_programs.append(program)
            accepted_urls.update(verified_urls_by_program.get(program_id, set()))

    candidates = list(structured_output.get("programs") or [])
    return {
        "verdict": "accepted" if not candidates or accepted_programs else "rejected",
        "release": not candidates or bool(accepted_programs),
        "candidate_count": len(candidates),
        "released_count": len(accepted_programs),
        "removed_count": len(candidates) - len(accepted_programs),
        "accepted_programs": accepted_programs,
        "accepted_program_ids": [str(item.get("id")) for item in accepted_programs],
        "accepted_urls": sorted(accepted_urls),
        "rejected_programs": rejected_programs,
        "checks": {
            "goal_identity": True,
            "online_discovery_binding": True,
            "official_evidence_binding": True,
            "official_url_binding": True,
        },
    }


def local_material_verification(
    payload: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic fallback used when no independent cloud verifier is available."""
    content = str(candidate.get("content") or "").strip()
    manifest = payload.get("context_manifest") or {}
    allowed_experiences = {
        str(item.get("id")) for item in manifest.get("experience_refs", []) if item.get("id")
    }
    cited_experiences = {
        str(value) for value in candidate.get("source_experience_ids", []) if value
    }
    issues = []
    if not content:
        issues.append("empty_content")
    if cited_experiences - allowed_experiences:
        issues.append("unsupported_experience_reference")
    if not manifest.get("official_evidence_refs"):
        issues.append("official_requirement_evidence_missing")
    return {
        "verdict": "accepted" if not issues else "rejected",
        "summary": "独立确定性校验通过" if not issues else "文稿未通过来源边界校验",
        "requirement_coverage": [],
        "unsupported_claims": [],
        "conflicts": [],
        "issues": issues,
        "confidence": 1.0,
        "mode": "deterministic",
    }


def verify_material_candidate(
    payload: Dict[str, Any],
    candidate: Dict[str, Any],
    semantic_report: Dict[str, Any],
) -> Dict[str, Any]:
    deterministic = local_material_verification(payload, candidate)
    semantic_verdict = str(semantic_report.get("verdict") or "rejected")
    issues = list(
        dict.fromkeys(
            [
                *deterministic.get("issues", []),
                *semantic_report.get("issues", []),
                *(
                    ["unsupported_claims"]
                    if semantic_report.get("unsupported_claims")
                    else []
                ),
                *(["source_conflicts"] if semantic_report.get("conflicts") else []),
            ]
        )
    )
    accepted = deterministic["verdict"] == "accepted" and semantic_verdict == "accepted"
    return {
        **semantic_report,
        "verdict": "accepted" if accepted else semantic_verdict if semantic_verdict != "accepted" else "rejected",
        "issues": issues,
        "deterministic_checks": {
            "source_boundary": deterministic["verdict"],
            "issues": deterministic.get("issues", []),
        },
    }
