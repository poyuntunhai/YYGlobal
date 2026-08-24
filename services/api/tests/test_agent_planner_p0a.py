from sqlalchemy import select

from app.agent.goals import parse_goal_spec, resolve_skill_and_goal
from app.agent.planner import build_plan
from app.agent.skills import skill_registry
from app.agent.tools import tool_registry
from app.agent.validators import (
    audit_program_research_run,
    validate_official_result,
    validate_program_search_result,
)
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import AgentRun, AgentStep, ToolCall


async def _new_program_run(session, goal_spec):
    skill = skill_registry.get("program-research")
    plan = build_plan(skill, "Rice University Computer Science", goal_spec)
    run = AgentRun(
        owner_id=settings.local_owner_id,
        skill_name=skill.name,
        skill_version=skill.version,
        goal="Rice University Computer Science",
        plan=plan,
    )
    session.add(run)
    await session.flush()
    for position, item in enumerate(plan):
        session.add(
            AgentStep(
                owner_id=settings.local_owner_id,
                run_id=run.id,
                position=position,
                name=item["name"],
                expected_output=item["expected_output"],
                checkpoint={
                    "plan_id": item["id"],
                    "step_type": item["step_type"],
                    "allowed_tools": item["allowed_tools"],
                },
                status="accepted" if position == 0 else "pending",
            )
        )
    await session.commit()
    return run


async def test_rice_goal_spec_and_catalog_results_enforce_identity_constraints(client):
    async with SessionLocal() as session:
        routed_skill, _ = await resolve_skill_and_goal(
            session, "Search Rice University Computer Science"
        )
        assert routed_skill.name == "program-research"
        goal_spec = await parse_goal_spec(
            session, "program-research", "Search Rice University Computer Science"
        )
        assert goal_spec["entities"]["university"] == "Rice University"
        assert goal_spec["entities"]["fields"] == ["computer science"]
        assert goal_spec["entities"]["degree_level"] == "master"

        accepted = await tool_registry.tools["search_programs"].handler(
            session,
            {"query": "Rice University Computer Science", "country": "", "field": ""},
        )
        evaluation = validate_program_search_result(goal_spec, accepted)
        assert evaluation["verdict"] == "accepted"
        assert accepted
        assert {item["university"] for item in accepted} == {"Rice University"}

        wrong_school = [{**accepted[0], "university": "Stanford University"}]
        rejected = validate_program_search_result(goal_spec, wrong_school)
        assert rejected["verdict"] == "rejected"
        assert rejected["reason_code"] == "hard_constraint_failed"


def test_program_search_validator_accepts_canonical_school_aliases():
    goal_spec = {
        "entities": {
            "university": "National University of Singapore (NUS)",
            "program": "",
            "degree_level": "master",
            "fields": ["computer science"],
            "countries": [],
        },
        "parameters": {"max_qs_rank": 100, "excluded_program_ids": []},
    }
    candidate = [{
        "id": "nus-mcs",
        "university": "National University of Singapore",
        "name": "Master of Computing in Computer Science",
        "degree": "master",
        "field": "Computer Science",
        "country": "Singapore",
        "qs_rank": 8,
    }]
    evaluation = validate_program_search_result(goal_spec, candidate)
    assert evaluation["verdict"] == "accepted"
    assert evaluation["accepted_program_ids"] == ["nus-mcs"]


def test_official_validator_requires_verbatim_evidence():
    rejected = validate_official_result(
        "verify_program_official",
        {"program_id": "p1", "official_url": "https://cs.rice.edu", "evidence": []},
    )
    assert rejected["verdict"] == "rejected"

    accepted = validate_official_result(
        "verify_program_official",
        {
            "program_id": "p1",
            "official_url": "https://cs.rice.edu",
            "status": "verified",
            "evidence": [{"field": "materials", "quote": "Submit a statement of purpose."}],
        },
    )
    assert accepted["verdict"] == "accepted"


async def test_tool_call_binds_to_recipe_step_and_final_auditor_blocks_unverified_program(
    client, monkeypatch
):
    async with SessionLocal() as session:
        goal_spec = await parse_goal_spec(
            session, "program-research", "Rice University Computer Science"
        )
        run = await _new_program_run(session, goal_spec)
        catalog_result = await tool_registry.tools["search_programs"].handler(
            session,
            {"query": "Rice University Computer Science", "country": "", "field": ""},
        )

        async def fake_discovery(_session, _arguments):
            return catalog_result

        monkeypatch.setattr(
            tool_registry.tools["discover_official_programs"], "handler", fake_discovery
        )
        result = await tool_registry.execute(
            session,
            run.id,
            "discover_official_programs",
            {
                "countries": [],
                "fields": ["Computer Science"],
                "degree_level": "master",
                "target_university": "Rice University",
                "max_qs_rank": 100,
                "excluded_program_ids": [],
                "max_candidates": 10,
            },
            ["discover_official_programs"],
            goal_spec=goal_spec,
        )
        trace = await session.scalar(
            select(ToolCall).where(ToolCall.run_id == run.id)
        )
        step = await session.get(AgentStep, trace.step_id)
        assert result
        assert trace.status == "completed"
        assert trace.result["_evaluation"]["verdict"] == "accepted"
        assert step.position == 1
        assert step.status == "accepted"

        candidate = result[0]
        structured = {
            "programs": [
                {
                    "id": candidate["id"],
                    "university": candidate["university"],
                    "name": candidate["name"],
                    "field": candidate["field"],
                    "official_url": candidate["official_url"],
                    "reason": "matched",
                }
            ],
            "sources": [],
            "unverified": [],
            "summary": "candidate",
        }
        audited, audit = await audit_program_research_run(session, run.id, structured)
        assert audit["release"] is False
        assert audited["programs"] == []
        assert "不发布未核验项目" in audited["summary"]


async def test_recipe_rejects_official_verification_before_search_step(client):
    async with SessionLocal() as session:
        goal_spec = await parse_goal_spec(
            session, "program-research", "Rice University Computer Science"
        )
        run = await _new_program_run(session, goal_spec)
        result = await tool_registry.execute(
            session,
            run.id,
            "verify_program_official",
            {"program_id": "not-used-before-dependency-check"},
            ["verify_program_official"],
            goal_spec=goal_spec,
        )
        trace = await session.scalar(
            select(ToolCall).where(ToolCall.run_id == run.id)
        )
        assert result["error"] == "hard_constraint_failed"
        assert result["evaluation"]["reason_code"] == "step_dependencies_not_satisfied"
        assert trace.status == "rejected"
        assert trace.result["_evaluation"]["verdict"] == "rejected"
