import asyncio
import time
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import build_plan
from app.agent.skills import skill_registry
from app.agent.tools import tool_registry
from app.agent.validators import audit_program_research_run, update_synthesis_steps
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import AgentRun, AgentStep, Program, RecommendationGoal
from app.services.business import canonical_countries, canonical_fields, get_program


def recommendation_goal_spec(
    goal: RecommendationGoal, excluded_program_ids: Set[str], limit: int
) -> Dict[str, Any]:
    # GoalSpec must retain every user-entered field.  Canonicalization is useful
    # as matching metadata, but using the canonical list as the goal itself
    # drops free-form fields whenever at least one sibling field is recognized.
    # Example: ["人工智能", "计算机科学", "国际教育"] previously became only
    # ["computer science"], so the independent validator could reject an
    # international-education result even though discovery searched for it.
    fields = list(
        dict.fromkeys(value.strip() for value in goal.target_fields if value and value.strip())
    )
    normalized_fields = canonical_fields(fields)
    target_university = goal.target_university.strip()
    countries = [] if target_university else canonical_countries(goal.target_countries)
    hard_constraints = [
        "指定学校优先于冲突的国家限制" if target_university else "国家或地区必须匹配",
        "QS 2026 排名不得超过设置上限" if goal.max_qs_rank else "QS 排名不限",
        "项目方向必须匹配目标方向",
        "学位层级必须匹配目标学位",
        "候选必须来自本次联网搜索并指向具体项目官网",
        "当前页面已展示项目不得重复",
    ]
    return {
        "recipe": "program_research",
        "intent": "recommend_programs",
        "entities": {
            "university": target_university,
            "program": "",
            "degree_level": goal.target_degree_level,
            "fields": fields,
            "countries": countries,
        },
        "parameters": {
            "max_qs_rank": goal.max_qs_rank,
            "excluded_program_ids": sorted(excluded_program_ids),
            "batch_size": limit,
            "canonical_fields": normalized_fields,
        },
        "hard_constraints": hard_constraints,
        "unknowns": [],
    }


async def _create_run(
    session: AsyncSession, goal: RecommendationGoal, goal_spec: Dict[str, Any]
) -> Tuple[AgentRun, List[AgentStep]]:
    skill = skill_registry.get("program-research")
    goal_text = (
        f"联网推荐 {goal.target_university or ', '.join(goal.target_countries)} "
        f"{', '.join(goal.target_fields)} {goal.target_degree_level} 项目"
    )
    plan = build_plan(skill, goal_text, goal_spec)
    run = AgentRun(
        owner_id=settings.local_owner_id,
        skill_name=skill.name,
        skill_version=skill.version,
        goal=goal_text,
        plan=plan,
        context_snapshot={
            "goal_spec": goal_spec,
            "recommendation_goal_id": goal.id,
            "discovery_mode": "live_web_only",
        },
        status="running",
    )
    session.add(run)
    await session.flush()
    steps: List[AgentStep] = []
    for position, item in enumerate(plan):
        step = AgentStep(
            owner_id=settings.local_owner_id,
            run_id=run.id,
            position=position,
            name=item["name"],
            expected_output=item["expected_output"],
            checkpoint={
                "plan_id": item["id"],
                "step_type": item.get("step_type", "generic"),
                "allowed_tools": item.get("allowed_tools", []),
                "acceptance_criteria": item.get("acceptance_criteria", []),
            },
            status="accepted" if position == 0 else "pending",
            result={"status": "accepted", "goal_spec": goal_spec} if position == 0 else {},
        )
        session.add(step)
        steps.append(step)
    await session.commit()
    return run, steps


async def run_online_program_recommendation(
    session: AsyncSession,
    goal: RecommendationGoal,
    *,
    excluded_program_ids: Set[str],
    limit: int = 5,
) -> Tuple[List[Program], AgentRun]:
    started = time.perf_counter()
    goal_spec = recommendation_goal_spec(goal, excluded_program_ids, limit)
    run, _ = await _create_run(session, goal, goal_spec)
    run_id = run.id
    try:
        candidates: List[Dict[str, Any]] = []
        candidate_ids: Set[str] = set()
        verified_ids: Set[str] = set()
        search_excluded_ids = set(excluded_program_ids)
        catalog_reports: List[Dict[str, Any]] = []
        calls_used = 0
        semaphore = asyncio.Semaphore(5)

        async def verify_candidate(item: Dict[str, Any]) -> Tuple[str, bool]:
            program_id = str(item.get("id") or "")
            if not program_id:
                return "", False
            async with semaphore, SessionLocal() as task_session:
                try:
                    result = await tool_registry.execute(
                        task_session,
                        run_id,
                        "verify_program_official",
                        {"program_id": program_id, "fast_mode": True},
                        ["verify_program_official"],
                        goal_spec=goal_spec,
                    )
                    evaluation = result.get("_evaluation", {}) if isinstance(result, dict) else {}
                    accepted = evaluation.get("verdict") == "accepted" or (
                        isinstance(result, dict)
                        and result.get("status") == "verified"
                        and bool(result.get("evidence"))
                    )
                    return program_id, accepted
                except Exception:
                    await task_session.rollback()
                    return program_id, False

        # A specified university can expose different valid programmes on
        # successive searches. Keep the exclusions accumulated within this
        # request and continue until the requested batch is full. Two
        # consecutive empty rounds are treated as exhaustion so a transiently
        # narrow result does not make a one-item batch look complete.
        max_rounds = min(max(1, limit), 5) if goal_spec["entities"]["university"] else 3
        consecutive_empty_rounds = 0
        for round_index in range(max_rounds):
            remaining = limit - len(verified_ids)
            if remaining <= 0:
                break
            reserve_for_another_discovery = 1 if round_index < max_rounds - 1 else 0
            verification_capacity = (
                settings.agent_max_tool_calls - calls_used - 1 - reserve_for_another_discovery
            )
            if verification_capacity <= 0:
                break
            discovery_arguments = {
                "countries": goal_spec["entities"]["countries"],
                "fields": list(goal.target_fields),
                "degree_level": goal.target_degree_level,
                "target_university": goal.target_university.strip(),
                "max_qs_rank": goal.max_qs_rank or 0,
                "excluded_program_ids": sorted(search_excluded_ids),
                "max_candidates": min(
                    verification_capacity,
                    max(remaining * 2, remaining + 2),
                ),
                "school_offset": 0 if goal.target_university.strip() else round_index * 6,
            }
            discovered = await tool_registry.execute(
                session,
                run_id,
                "discover_official_programs",
                discovery_arguments,
                ["discover_official_programs"],
                goal_spec=goal_spec,
            )
            calls_used += 1
            if isinstance(discovered, dict) and isinstance(discovered.get("items"), list):
                discovered_items = discovered["items"]
                report = discovered.get("catalog_report")
                if isinstance(report, dict):
                    catalog_reports.append(report)
            else:
                discovered_items = discovered if isinstance(discovered, list) else []
            new_candidates = []
            for item in discovered_items:
                program_id = str(item.get("id") or "")
                if not program_id or program_id in candidate_ids:
                    continue
                candidate_ids.add(program_id)
                search_excluded_ids.add(program_id)
                candidates.append(item)
                new_candidates.append(item)
            if not new_candidates:
                consecutive_empty_rounds += 1
                if consecutive_empty_rounds >= 2:
                    break
                continue
            consecutive_empty_rounds = 0
            available_calls = settings.agent_max_tool_calls - calls_used
            if round_index < max_rounds - 1:
                available_calls -= 1
            to_verify = new_candidates[: max(0, available_calls)]
            verification_results = await asyncio.gather(
                *(verify_candidate(item) for item in to_verify)
            )
            calls_used += len(to_verify)
            verified_ids.update(
                program_id for program_id, accepted in verification_results if accepted
            )
        session.expire_all()
        await session.refresh(run)

        structured_output = {
            "programs": [
                {
                    "id": item.get("id"),
                    "university": item.get("university"),
                    "name": item.get("name"),
                    "degree": item.get("degree"),
                    "country": item.get("country"),
                    "field": item.get("field"),
                    "qs_rank": item.get("qs_rank"),
                    "official_url": item.get("official_url"),
                    "catalog_url": item.get("catalog_url"),
                    "faculty_catalog_url": item.get("faculty_catalog_url"),
                    "reason": "本次联网发现并进入官网核验",
                }
                for item in candidates
            ],
            "sources": [
                {"url": item.get("official_url"), "program_id": item.get("id")}
                for item in candidates
                if item.get("official_url")
            ],
            "unverified": [],
            "summary": f"本次联网发现 {len(candidates)} 个候选项目。",
            "discovery_report": {
                "mode": "official_catalog_first",
                "target_count": limit,
                "catalog_rounds": catalog_reports,
                "candidate_count": len(candidates),
                "verified_count": len(verified_ids),
                "catalogs_exhausted": (
                    bool(goal_spec["entities"]["university"])
                    and len(catalog_reports) == max_rounds
                    and all(report.get("status") == "completed" for report in catalog_reports)
                    and all(
                        any(
                            school.get("catalogs")
                            for school in report.get("schools", [])
                            if isinstance(school, dict)
                        )
                        for report in catalog_reports
                    )
                    and len(verified_ids) < limit
                )
                or (
                    not goal_spec["entities"]["university"]
                    and bool(catalog_reports)
                    and catalog_reports[-1].get("status") == "no_candidate_school"
                ),
            },
        }
        audited, audit = await audit_program_research_run(
            session, run_id, structured_output, goal_spec
        )
        steps = await update_synthesis_steps(session, run_id, audit)
        accepted_ids = [
            str(item.get("id")) for item in audited.get("programs", []) if item.get("id")
        ][:limit]
        programs = [
            program
            for program_id in accepted_ids
            if (program := await get_program(session, program_id)) is not None
        ]
        statuses = {step.position: step.status for step in steps}
        run.plan = [
            {**item, "status": statuses.get(index, item.get("status", "pending"))}
            for index, item in enumerate(run.plan or [])
        ]
        run.status = "completed"
        run.stop_reason = "success" if programs else "verification_incomplete"
        run.final_output = (
            f"已发布 {len(programs)} 个通过官网证据审计的新项目。"
            if programs
            else "本次联网搜索未发现通过完整官网证据审计的新项目。"
        )
        run.structured_output = audited
        run.duration_ms = int((time.perf_counter() - started) * 1000)
        await session.commit()
        return programs, run
    except asyncio.CancelledError:
        run.status = "cancelled"
        run.stop_reason = "client_cancelled"
        run.final_output = "项目检索请求已取消。"
        run.duration_ms = int((time.perf_counter() - started) * 1000)
        await asyncio.shield(session.commit())
        raise
    except Exception as exc:
        run.status = "failed"
        run.stop_reason = "error"
        run.final_output = str(exc)[:1000]
        run.duration_ms = int((time.perf_counter() - started) * 1000)
        await session.commit()
        raise
