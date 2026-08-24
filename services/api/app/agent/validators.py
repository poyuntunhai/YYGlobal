from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.verifier import verify_program_research_release
from app.models.entities import AgentStep, ToolCall
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


def _field_matches(actual: Any, expected: Sequence[str]) -> bool:
    if not expected:
        return True
    actual_fields = set(canonical_fields([str(actual or "")]))
    expected_fields = set(canonical_fields([str(value) for value in expected]))
    if actual_fields and expected_fields:
        return bool(actual_fields & expected_fields)
    normalized_actual = _normalized(actual)
    return any(
        _normalized(value) in normalized_actual or normalized_actual in _normalized(value)
        for value in expected
        if _normalized(value)
    )


def validate_program_search_result(
    goal_spec: Dict[str, Any], result: Any
) -> Dict[str, Any]:
    entities = goal_spec.get("entities", {})
    university = str(entities.get("university") or "")
    program = _normalized(entities.get("program"))
    degree = str(entities.get("degree_level") or "")
    fields = list(entities.get("fields") or [])
    countries = {_normalized(value) for value in entities.get("countries") or []}
    parameters = goal_spec.get("parameters", {})
    max_qs_rank = parameters.get("max_qs_rank")
    excluded_ids = {str(value) for value in parameters.get("excluded_program_ids") or []}
    rows = (
        result
        if isinstance(result, list)
        else list(result.get("items") or [])
        if isinstance(result, dict)
        else []
    )
    accepted = []
    rejected = []
    for row in rows:
        reasons = []
        if university and not same_university(str(row.get("university") or ""), university):
            reasons.append("wrong_university")
        if program and _normalized(row.get("name")) != program:
            reasons.append("wrong_program")
        if degree and not _degree_matches(row.get("degree"), degree):
            reasons.append("wrong_degree_level")
        if fields and not _field_matches(row.get("field"), fields):
            reasons.append("wrong_field")
        if countries and _normalized(row.get("country")) not in countries:
            reasons.append("wrong_country")
        rank = row.get("qs_rank")
        if max_qs_rank and (
            not isinstance(rank, (int, float)) or rank < 1 or rank > max_qs_rank
        ):
            reasons.append("outside_qs_range")
        if str(row.get("id") or "") in excluded_ids:
            reasons.append("already_displayed")
        if reasons:
            rejected.append({"id": row.get("id"), "reasons": reasons})
        else:
            accepted.append(row)
    if not accepted:
        return {
            "verdict": "rejected",
            "reason_code": "hard_constraint_failed",
            "message": "项目搜索结果没有满足学校、项目、专业或学位层级硬约束。",
            "accepted_count": 0,
            "rejected": rejected,
        }
    return {
        "verdict": "accepted",
        "reason_code": "identity_constraints_satisfied",
        "message": "项目身份和硬约束已通过验收。",
        "accepted_count": len(accepted),
        "accepted_program_ids": [row.get("id") for row in accepted if row.get("id")],
        "rejected": rejected,
    }


def validate_official_result(tool_name: str, result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "verdict": "rejected",
            "reason_code": "invalid_official_result",
            "message": "官网核验工具没有返回结构化结果。",
        }
    if tool_name == "fetch_official_source":
        url = str(result.get("url") or "")
        hostname = (urlparse(url).hostname or "").lower()
        if not hostname or result.get("status") in {"failed", "error"}:
            return {
                "verdict": "rejected",
                "reason_code": "official_source_unavailable",
                "message": "没有取得可用的官方页面。",
            }
        return {
            "verdict": "progress",
            "reason_code": "official_source_fetched",
            "message": "已取得官网页面，仍需提取并验收逐字证据。",
        }
    evidence = list(result.get("evidence") or [])
    source_results = list(result.get("source_results") or [])
    has_official_source = bool(
        result.get("official_url")
        or result.get("source_id")
        or any(item.get("url") for item in source_results if isinstance(item, dict))
    )
    valid_quotes = [
        item for item in evidence
        if isinstance(item, dict) and str(item.get("quote") or "").strip()
    ]
    verification_complete = result.get("status") == "verified"
    if not has_official_source or not valid_quotes or not verification_complete:
        return {
            "verdict": "rejected",
            "reason_code": "insufficient_official_evidence",
            "message": "官网结果缺少可用的招生核心字段逐字证据，不能标记为已核验。",
            "evidence_count": len(valid_quotes),
        }
    return {
        "verdict": "accepted",
        "reason_code": "official_evidence_verified",
        "message": "官网来源和逐字证据已通过验收。",
        "evidence_count": len(valid_quotes),
        "program_id": result.get("program_id"),
    }


def evaluate_tool_result(
    goal_spec: Dict[str, Any], tool_name: str, result: Any
) -> Dict[str, Any]:
    if not goal_spec or goal_spec.get("recipe") != "program_research":
        return {"verdict": "accepted", "reason_code": "not_applicable", "message": ""}
    if tool_name in {
        "discover_official_programs", "search_programs", "mcp_catalog_search"
    }:
        return validate_program_search_result(goal_spec, result)
    if tool_name in {
        "fetch_official_source",
        "extract_program_requirements",
        "verify_program_official",
    }:
        return validate_official_result(tool_name, result)
    return {"verdict": "accepted", "reason_code": "not_applicable", "message": ""}


async def audit_program_research_run(
    session: AsyncSession,
    run_id: str,
    structured_output: Dict[str, Any],
    goal_spec: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    calls = list(
        (
            await session.scalars(
                select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.created_at)
            )
        ).all()
    )
    report = verify_program_research_release(goal_spec or {}, structured_output, calls)
    programs = list(structured_output.get("programs") or [])
    released_programs = report["accepted_programs"]
    removed_count = report["removed_count"]
    output = dict(structured_output)
    output["programs"] = released_programs
    output["sources"] = [
        item
        for item in (output.get("sources") or [])
        if isinstance(item, dict)
        and str(item.get("url") or "") in set(report["accepted_urls"])
    ]
    unverified = list(output.get("unverified") or [])
    if removed_count:
        unverified.append(
            f"{removed_count} 个候选项目未完成官网逐字证据验收，未作为已核验结果展示"
        )
    if programs and not released_programs:
        output["summary"] = "候选项目尚未完成官网逐字证据验收，本次不发布未核验项目。"
    output["unverified"] = list(dict.fromkeys(unverified))
    output["verification_report"] = {
        key: value for key, value in report.items() if key != "accepted_programs"
    }
    audit = {
        **output["verification_report"],
        "removed_unverified_count": removed_count,
        "verified_program_ids": report["accepted_program_ids"],
    }
    return output, audit


async def update_synthesis_steps(
    session: AsyncSession,
    run_id: str,
    audit: Dict[str, Any],
) -> List[AgentStep]:
    steps = list(
        (
            await session.scalars(
                select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.position)
            )
        ).all()
    )
    for step in steps:
        step_type = str((step.checkpoint or {}).get("step_type") or "")
        if step_type == "parse_search_goal" and step.status == "pending":
            step.status = "accepted"
            step.result = {"status": "accepted", "reason": "goal_spec_created"}
        elif step_type == "independent_verify_result":
            step.status = "accepted" if audit.get("release") else "rejected"
            step.result = {"status": step.status, "verification_report": audit}
        elif step_type in {"match_applicant_profile", "render_recommendations"}:
            step.status = "accepted" if audit.get("release") else "blocked"
            step.result = {"status": step.status, "audit": audit}
    await session.flush()
    return steps
